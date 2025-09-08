import concurrent.futures
import time
import random
import requests
import json
import psutil
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
from collections import deque
import threading
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class TestResult:
    """개별 테스트 결과"""
    user_id: int
    pdf_file: str
    pdf_pages: int
    pdf_size_mb: float
    start_time: float
    end_time: float
    total_duration: float
    ocr_duration: float
    translation_duration: float
    success: bool
    error: str = None
    workflow_id: str = None
    cpu_peak: float = 0
    memory_peak_mb: float = 0


class RealWorldStressTester:
    def __init__(self, pdf_folder: str, base_url: str = "http://localhost:8000"):
        self.pdf_folder = Path(pdf_folder)
        self.base_url = base_url.rstrip("/")
        self.test_users = []
        self.results: List[TestResult] = []
        self.fail_details: List[Dict] = []

        # 로깅 설정 (최우선 세팅)
        log_filename = f"stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # PDF 파일 로드는 로거 준비 이후 수행
        self.pdf_files = self._load_pdf_files()

        # 쿠키 도메인
        self._cookie_domain = urlparse(self.base_url).hostname

        # 랜덤 유니크 분배용 풀과 락
        self._pool_lock = threading.Lock()
        self._shuffled_pool: deque = deque()
        # 서버 최대 동시 워크플로 상한을 캐시하기 위한 값(옵션)
        self.server_max_concurrent: int | None = None

    def _load_pdf_files(self) -> List[Dict]:
        """PDF 파일 정보 로드"""
        pdf_files = []

        for pdf_path in self.pdf_folder.glob("*.pdf"):
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(str(pdf_path))
                pdf_info = {
                    'path': pdf_path,
                    'name': pdf_path.name,
                    'pages': len(doc),
                    'size_mb': pdf_path.stat().st_size / (1024 * 1024)
                }
                doc.close()
                pdf_files.append(pdf_info)
                self.logger.info(f"Loaded PDF: {pdf_info['name']} ({pdf_info['pages']} pages, {pdf_info['size_mb']:.2f} MB)")
            except Exception as e:
                self.logger.error(f"Failed to load {pdf_path}: {e}")

        # 크기별로 정렬
        pdf_files.sort(key=lambda x: x['pages'])

        self.logger.info(f"Total PDFs loaded: {len(pdf_files)}")
        return pdf_files

    def _pick_random_unique(self) -> Dict:
        """중복을 최소화하기 위해 셔플된 풀에서 하나씩 꺼내기"""
        with self._pool_lock:
            if not self._shuffled_pool:
                pool = list(self.pdf_files)
                random.shuffle(pool)
                self._shuffled_pool = deque(pool)
            return self._shuffled_pool.popleft()

    def setup_test_users(self, count: int):
        """테스트용 사용자 생성 및 로그인"""
        self.logger.info(f"Setting up {count} test users...")

        for i in range(count):
            username = f"stress_user_{i:04d}"
            email = f"stress{i:04d}@test.com"
            password = "StressTest123!"

            # 회원가입 시도 (실패해도 무시: 이미 존재 가능)
            try:
                requests.post(
                    f"{self.base_url}/api/v1/auth/register",
                    json={
                        "username": username,
                        "email": email,
                        "password": password
                    },
                    timeout=10
                )
            except Exception:
                pass

            # 로그인
            try:
                session = requests.Session()
                response = session.post(
                    f"{self.base_url}/api/v1/auth/login",
                    json={
                        "username": username,
                        "password": password
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    session_token = response.cookies.get('session_token')
                    # 세션과 세션 토큰을 함께 보관하여 업로드에도 동일 세션 사용
                    self.test_users.append({
                        'id': i,
                        'username': username,
                        'session_token': session_token,
                        'session': session
                    })
                    self.logger.info(f"User ready: {username}")
                else:
                    self.logger.error(f"Login failed for {username}: {response.status_code} {response.text}")
            except Exception as e:
                self.logger.error(f"Failed to setup user {username}: {e}")

    def process_single_document(self, user_info: Dict, pdf_info: Dict) -> TestResult:
        """단일 문서 처리 (전체 파이프라인)"""
        start_time = time.time()
        # 로그인에 사용한 동일한 세션을 우선 사용
        session = user_info.get('session') or requests.Session()
        # 세션 토큰이 있고 새 세션인 경우 쿠키/헤더를 보강
        if user_info.get('session_token') and 'session' not in user_info:
            token = user_info['session_token']
            try:
                session.cookies.set('session_token', token, domain=self._cookie_domain)
            except Exception:
                session.cookies.set('session_token', token)
            # 헤더에도 직접 추가(보수적 처리)
            session.headers['Cookie'] = f"session_token={token}"

        # 리소스 모니터링 시작 (클라이언트 프로세스 기준)
        process = psutil.Process()
        cpu_peak = 0
        memory_peak_mb = 0

        try:
            # 1. PDF 업로드
            self.logger.info(f"User {user_info['id']}: Uploading {pdf_info['name']}")

            # 업로드 요청 with 재시도(혼잡 시 429 처리)
            def _do_upload() -> requests.Response:
                with open(pdf_info['path'], 'rb') as f:
                    files = {'file': (pdf_info['name'], f, 'application/pdf')}
                    return session.post(
                        f"{self.base_url}/api/v1/documents/upload",
                        files=files,
                        timeout=120
                    )

            # 429 시: 슬롯이 빌 때까지 데드라인까지 대기
            start_upload_wait = time.time()
            upload_deadline = start_upload_wait + getattr(self, 'upload_wait_seconds', 600)
            attempt = 0
            backoff = 1.0
            while True:
                upload_response = _do_upload()
                if upload_response.status_code == 429:
                    attempt += 1
                    now = time.time()
                    if now >= upload_deadline:
                        break
                    retry_after = upload_response.headers.get('Retry-After')
                    if retry_after:
                        try:
                            sleep_s = float(retry_after)
                        except Exception:
                            sleep_s = backoff + random.uniform(0, 0.5)
                    else:
                        sleep_s = backoff + random.uniform(0, 0.5)
                    # 데드라인 초과하지 않도록 슬립 캡
                    if now + sleep_s > upload_deadline:
                        sleep_s = max(0.2, upload_deadline - now)
                    self.logger.warning(f"User {user_info['id']}: Server busy (429). waiting {sleep_s:.1f}s until slot free (attempt {attempt})")
                    time.sleep(sleep_s)
                    backoff = min(backoff * 2, 8.0)
                    continue
                break

            if upload_response.status_code != 200:
                raise Exception(f"Upload failed: {upload_response.status_code} {upload_response.text}")

            data = upload_response.json()
            workflow_id = data.get('workflow_id')
            if not workflow_id:
                raise Exception("workflow_id missing in upload response")

            upload_time = time.time()

            # 2. 진행 상황 모니터링
            self.logger.info(f"User {user_info['id']}: Processing workflow {workflow_id}")

            ocr_complete = False
            ocr_duration = 0.0
            translation_duration = 0.0
            check_interval = 2.0
            last_logged_progress = -1

            # 단계별 타임아웃 구성
            ocr_max_wait = getattr(self, 'ocr_max_wait_seconds', 900)
            translation_max_wait = getattr(self, 'translation_max_wait_seconds', 900)
            ocr_deadline = upload_time + ocr_max_wait
            translation_start_time = None

            last_status_snapshot = None
            while True:
                # 리소스 체크 (짧은 인터벌로 샘플링 정확도 확보)
                cpu_current = process.cpu_percent(interval=0.05)
                memory_current = process.memory_info().rss / (1024 * 1024)
                cpu_peak = max(cpu_peak, cpu_current)
                memory_peak_mb = max(memory_peak_mb, memory_current)

                # 상태 확인
                try:
                    status_response = session.get(
                        f"{self.base_url}/api/v1/workflows/{workflow_id}",
                        timeout=30
                    )
                except requests.exceptions.ReadTimeout:
                    # 일시적 지연은 실패로 처리하지 않고 다음 폴링으로
                    time.sleep(1.0)
                    continue

                if status_response.status_code == 200:
                    status_data = status_response.json()
                    last_status_snapshot = status_data
                    current_stage = status_data.get('current_stage')
                    status = status_data.get('status')
                    progress = status_data.get('progress_percentage', 0) or 0

                    # OCR 완료 시점 기록
                    if not ocr_complete and current_stage in ['TRANSLATION', 'COMPLETION']:
                        ocr_complete = True
                        ocr_duration = time.time() - upload_time
                        translation_start_time = time.time()
                        self.logger.info(f"User {user_info['id']}: OCR completed in {ocr_duration:.2f}s")

                    # 완료 확인
                    if status == 'COMPLETED':
                        # 번역 시작 시간이 기록되었다면 그것을 기준으로 계산
                        if translation_start_time:
                            translation_duration = time.time() - translation_start_time
                        else:
                            translation_duration = time.time() - upload_time - ocr_duration
                        self.logger.info(f"User {user_info['id']}: Translation completed in {translation_duration:.2f}s")
                        break
                    elif status == 'FAILED':
                        raise Exception(f"Workflow failed: {status_data.get('error_info')}")

                    # 진행률 로그 (10% 단위, 변화 시에만)
                    if isinstance(progress, int):
                        bucket = (progress // 10) * 10
                        if bucket != last_logged_progress:
                            last_logged_progress = bucket
                            self.logger.debug(f"User {user_info['id']}: {bucket}% complete")

                # 단계별 타임아웃 검사
                now_time = time.time()
                if not ocr_complete and now_time > ocr_deadline:
                    raise Exception("Timeout(OCR) waiting for completion")
                if ocr_complete:
                    # 번역 타임아웃은 번역 단계 진입 시점부터 계산
                    trans_start = translation_start_time or now_time
                    if now_time > trans_start + translation_max_wait:
                        raise Exception("Timeout(TRANSLATION) waiting for completion")

                # 지터가 있는 대기 (스파이크 완화)
                sleep_s = check_interval + random.uniform(-0.3, 0.3)
                if sleep_s < 0.2:
                    sleep_s = 0.2
                time.sleep(sleep_s)

            # 이 루프는 완료 또는 예외로만 탈출함

            # 3. 성공
            total_duration = time.time() - start_time

            result = TestResult(
                user_id=user_info['id'],
                pdf_file=pdf_info['name'],
                pdf_pages=pdf_info['pages'],
                pdf_size_mb=pdf_info['size_mb'],
                start_time=start_time,
                end_time=time.time(),
                total_duration=total_duration,
                ocr_duration=ocr_duration,
                translation_duration=translation_duration,
                success=True,
                workflow_id=workflow_id,
                cpu_peak=cpu_peak,
                memory_peak_mb=memory_peak_mb
            )

            self.logger.info(
                f"User {user_info['id']}: SUCCESS - Total {total_duration:.2f}s (OCR: {ocr_duration:.2f}s, Translation: {translation_duration:.2f}s)"
            )
            return result

        except Exception as e:
            total_duration = time.time() - start_time
            self.logger.error(f"User {user_info['id']}: FAILED - {str(e)}")
            # 상태 조회 시도 (가능하면 워크플로 상태 추가)
            try:
                if 'workflow_id' in locals():
                    status_response = session.get(f"{self.base_url}/api/v1/workflows/{workflow_id}", timeout=10)
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        self.fail_details.append({
                            'user_id': user_info['id'],
                            'pdf': pdf_info['name'],
                            'reason': str(e),
                            'workflow_id': workflow_id,
                            'last_status': status_data
                        })
            except Exception:
                pass

            return TestResult(
                user_id=user_info['id'],
                pdf_file=pdf_info['name'],
                pdf_pages=pdf_info['pages'],
                pdf_size_mb=pdf_info['size_mb'],
                start_time=start_time,
                end_time=time.time(),
                total_duration=total_duration,
                ocr_duration=0.0,
                translation_duration=0.0,
                success=False,
                error=str(e),
                cpu_peak=cpu_peak,
                memory_peak_mb=memory_peak_mb
            )

    def run_concurrent_test(
        self,
        concurrent_users: int,
        pdf_distribution: str = 'random',
        duration_seconds: int = None,
        total_requests: int = None,
        per_user_once: bool = False
    ) -> Dict:
        """동시 사용자 테스트 실행"""
        self.logger.info("=" * 60)
        self.logger.info(f"Starting concurrent test: {concurrent_users} users")
        self.logger.info(f"PDF distribution: {pdf_distribution}")
        self.logger.info("=" * 60)

        # PDF 선택 전략
        def select_pdf(index: int) -> Dict:
            if not self.pdf_files:
                raise RuntimeError("No PDF files loaded")
            n = len(self.pdf_files)
            if pdf_distribution == 'sequential':
                return self.pdf_files[index % n]
            elif pdf_distribution == 'random_unique':
                return self._pick_random_unique()
            elif pdf_distribution == 'weighted' and n >= 3:
                # 작은 파일 70%, 중간 20%, 큰 파일 10%
                rand = random.random()
                first = n // 3 or 1
                second = (2 * n) // 3 or (first + 1)
                if rand < 0.7:
                    return random.choice(self.pdf_files[:first])
                elif rand < 0.9:
                    return random.choice(self.pdf_files[first:second])
                else:
                    return random.choice(self.pdf_files[second:])
            else:  # random
                return random.choice(self.pdf_files)

        # 시스템 리소스 모니터링 시작
        initial_cpu = psutil.cpu_percent(interval=1)
        initial_memory = psutil.virtual_memory().percent

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures: List[concurrent.futures.Future] = []
            request_count = 0
            used_user_ids = set()
            exhausted_users = False

            def _prune_done():
                # 완료된 future 제거하여 큐 길이 유지
                alive = []
                for f in futures:
                    if f.done():
                        try:
                            res = f.result(timeout=0)
                            self.results.append(res)
                        except Exception as err:
                            self.logger.error(f"Task failed with exception: {err}")
                    else:
                        alive.append(f)
                futures[:] = alive

            # 요청 생성
            if duration_seconds:
                # 시간 기반
                while time.time() - start_time < duration_seconds:
                    for _ in range(concurrent_users):
                        _prune_done()
                        if len(futures) < concurrent_users * 2:  # 큐 제한
                            if per_user_once:
                                # 아직 제출하지 않은 사용자 선택
                                user = None
                                for candidate in self.test_users:
                                    if candidate['id'] not in used_user_ids:
                                        user = candidate
                                        break
                                if user is None:
                                    exhausted_users = True
                                    break
                                used_user_ids.add(user['id'])
                            else:
                                user = self.test_users[request_count % len(self.test_users)]
                            pdf = select_pdf(request_count)
                            future = executor.submit(self.process_single_document, user, pdf)
                            futures.append(future)
                            request_count += 1
                    # 램프업 + 지터
                    time.sleep(0.5 + random.uniform(-0.1, 0.1))
                    if exhausted_users and not futures:
                        break
            else:
                # 횟수 기반
                total_requests = total_requests or concurrent_users
                max_reqs = total_requests
                if per_user_once:
                    max_reqs = min(total_requests, len(self.test_users))
                for i in range(max_reqs):
                    _prune_done()
                    if per_user_once:
                        user = self.test_users[i]
                        used_user_ids.add(user['id'])
                    else:
                        user = self.test_users[i % len(self.test_users)]
                    pdf = select_pdf(i)
                    future = executor.submit(self.process_single_document, user, pdf)
                    futures.append(future)

                    # 동시 실행 제한
                    if (i + 1) % concurrent_users == 0:
                        time.sleep(1)

            # 남은 결과 수집
            for future in concurrent.futures.as_completed(futures, timeout=None):
                try:
                    # 사용자별 합산 타임아웃: 업로드 대기 + OCR + 번역 + 여유
                    per_task_timeout = (
                        getattr(self, 'upload_wait_seconds', 600)
                        + getattr(self, 'ocr_max_wait_seconds', 900)
                        + getattr(self, 'translation_max_wait_seconds', 900)
                        + 120
                    )
                    result = future.result(timeout=per_task_timeout)
                    self.results.append(result)
                except Exception as e:
                    self.logger.error(f"Task failed with exception: {e}")

        # 시스템 리소스 최종
        final_cpu = psutil.cpu_percent(interval=1)
        final_memory = psutil.virtual_memory().percent

        # 결과 분석
        analysis = self.analyze_results()
        analysis['system_impact'] = {
            'cpu_increase': final_cpu - initial_cpu,
            'memory_increase': final_memory - initial_memory,
            'peak_cpu': max((r.cpu_peak for r in self.results), default=0),
            'peak_memory_mb': max((r.memory_peak_mb for r in self.results), default=0)
        }

        return analysis

    def analyze_results(self) -> Dict:
        """결과 분석"""
        if not self.results:
            return {}

        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        # 기본 통계
        stats = {
            'total_requests': len(self.results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': (len(successful) / len(self.results) * 100) if self.results else 0
        }

        if successful:
            # 시간 통계
            total_times = [r.total_duration for r in successful]
            ocr_times = [r.ocr_duration for r in successful if r.ocr_duration > 0]
            trans_times = [r.translation_duration for r in successful if r.translation_duration > 0]

            total_sorted = sorted(total_times)
            n = len(total_sorted)
            stats['time_stats'] = {
                'total': {
                    'mean': sum(total_times) / n,
                    'min': total_sorted[0],
                    'max': total_sorted[-1],
                    'p50': total_sorted[n // 2],
                    'p90': total_sorted[int(n * 0.9) if n > 0 else 0],
                    'p99': total_sorted[int(n * 0.99)] if n > 100 else total_sorted[-1]
                }
            }

            if ocr_times:
                stats['time_stats']['ocr'] = {
                    'mean': sum(ocr_times) / len(ocr_times),
                    'min': min(ocr_times),
                    'max': max(ocr_times)
                }

            if trans_times:
                stats['time_stats']['translation'] = {
                    'mean': sum(trans_times) / len(trans_times),
                    'min': min(trans_times),
                    'max': max(trans_times)
                }

            # 페이지별 성능
            pages_groups: Dict[str, List[float]] = {}
            for r in successful:
                if r.pdf_pages <= 5:
                    group = 'small'
                elif r.pdf_pages <= 20:
                    group = 'medium'
                else:
                    group = 'large'
                pages_groups.setdefault(group, []).append(r.total_duration)

            stats['performance_by_size'] = {
                group: {
                    'count': len(times),
                    'avg_time': sum(times) / len(times)
                }
                for group, times in pages_groups.items()
            }

        # 에러 분석
        if failed:
            error_types: Dict[str, int] = {}
            for r in failed:
                error_msg = (r.error or 'Unknown')
                error_msg = error_msg[:50]
                error_types[error_msg] = error_types.get(error_msg, 0) + 1
            stats['errors'] = error_types

        return stats

    def generate_report(self):
        """상세 리포트 생성"""
        if not self.results:
            self.logger.warning("No results to report")
            return

        # DataFrame 생성
        df = pd.DataFrame([asdict(r) for r in self.results])

        # 시각화
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. 성공/실패 비율
        success_counts = df['success'].value_counts()
        labels = ['Failed', 'Success'] if False in success_counts.index else ['Success']
        values = [success_counts.get(False, 0), success_counts.get(True, 0)]
        axes[0, 0].pie(values, labels=labels, autopct='%1.1f%%', colors=['red', 'green'])
        axes[0, 0].set_title('Success Rate')

        # 2. 처리 시간 분포
        success_mask = df['success'] == True
        if success_mask.any():
            axes[0, 1].hist(df.loc[success_mask, 'total_duration'], bins=20, color='blue', alpha=0.7)
            axes[0, 1].set_xlabel('Total Duration (seconds)')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Processing Time Distribution')

        # 3. OCR vs Translation 시간
        if 'ocr_duration' in df.columns and (df['ocr_duration'] > 0).any():
            ocr_avg = df.loc[df['ocr_duration'] > 0, 'ocr_duration'].mean()
            trans_avg = df.loc[df['translation_duration'] > 0, 'translation_duration'].mean()
            axes[0, 2].bar(['OCR', 'Translation'], [ocr_avg, trans_avg], color=['orange', 'purple'])
            axes[0, 2].set_ylabel('Average Time (seconds)')
            axes[0, 2].set_title('OCR vs Translation Time')

        # 4. 페이지 수별 처리 시간
        if success_mask.any():
            axes[1, 0].scatter(df.loc[success_mask, 'pdf_pages'], df.loc[success_mask, 'total_duration'], alpha=0.5)
            axes[1, 0].set_xlabel('PDF Pages')
            axes[1, 0].set_ylabel('Total Duration (seconds)')
            axes[1, 0].set_title('Processing Time by PDF Size')

        # 5. 시간대별 성공률
        df['hour'] = pd.to_datetime(df['start_time'], unit='s').dt.hour
        hourly_success = df.groupby('hour')['success'].mean() * 100
        if not hourly_success.empty:
            axes[1, 1].plot(hourly_success.index, hourly_success.values, marker='o')
            axes[1, 1].set_xlabel('Hour')
            axes[1, 1].set_ylabel('Success Rate (%)')
            axes[1, 1].set_title('Success Rate by Hour')

        # 6. 리소스 사용량
        if (df['memory_peak_mb'] > 0).any() or (df['cpu_peak'] > 0).any():
            axes[1, 2].boxplot([
                df['cpu_peak'].dropna(),
                (df['memory_peak_mb'].dropna() / 100)
            ], tick_labels=['CPU (%)', 'Memory (GB/100)'])
            axes[1, 2].set_title('Resource Usage')

        plt.tight_layout()

        # 파일 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'stress_test_report_{timestamp}.png', dpi=100, bbox_inches='tight')
        df.to_csv(f'stress_test_results_{timestamp}.csv', index=False)

        # 실패 상세 저장
        if self.fail_details:
            import csv
            with open(f'stress_test_failures_{timestamp}.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['user_id', 'pdf', 'reason', 'workflow_id', 'status', 'stage', 'progress'])
                for item in self.fail_details:
                    status = item.get('last_status') or {}
                    writer.writerow([
                        item.get('user_id'),
                        item.get('pdf'),
                        item.get('reason'),
                        item.get('workflow_id'),
                        status.get('status'),
                        status.get('current_stage'),
                        status.get('progress_percentage')
                    ])

        # 텍스트 리포트
        with open(f'stress_test_summary_{timestamp}.txt', 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("STRESS TEST SUMMARY REPORT\n")
            f.write("="*60 + "\n\n")

            analysis = self.analyze_results()
            f.write(json.dumps(analysis, indent=2, ensure_ascii=False))

            f.write("\n\n" + "="*60 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*60 + "\n")

            # 권장사항
            success_rate = analysis.get('success_rate', 0)
            if success_rate >= 95:
                f.write("✅ 시스템이 안정적입니다.\n")
            elif success_rate >= 80:
                f.write("⚠️ 일부 개선이 필요합니다.\n")
            else:
                f.write("❌ 심각한 성능 문제가 있습니다.\n")

            if 'time_stats' in analysis:
                avg_time = analysis['time_stats']['total']['mean']
                if avg_time > 300:  # 5분 이상
                    f.write("- 처리 시간이 너무 깁니다. 최적화가 필요합니다.\n")
                if analysis['time_stats']['total']['p90'] > avg_time * 2:
                    f.write("- 처리 시간 편차가 큽니다. 일관성 개선이 필요합니다.\n")

        plt.show()
        self.logger.info(f"Report saved: stress_test_report_{timestamp}.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Real World Stress Test for Document Translation')
    parser.add_argument('pdf_folder', help='Folder containing PDF files')
    parser.add_argument('--users', type=int, default=10, help='Number of concurrent users')
    parser.add_argument('--duration', type=int, help='Test duration in seconds')
    parser.add_argument('--requests', type=int, help='Total number of requests')
    parser.add_argument('--distribution', choices=['random', 'sequential', 'weighted', 'random_unique'],
                        default='random', help='PDF selection strategy')
    parser.add_argument('--server', default='http://localhost:8000', help='Server URL')
    parser.add_argument('--per-user-once', action='store_true', help='Each user uploads at most once')
    parser.add_argument('--upload-wait-seconds', type=int, default=600, help='Max seconds to wait for an upload slot when server is busy (429)')
    parser.add_argument('--ocr-max-wait-seconds', type=int, default=900, help='Max seconds to wait for OCR stage per workflow')
    parser.add_argument('--translation-max-wait-seconds', type=int, default=900, help='Max seconds to wait for TRANSLATION stage per workflow')
    parser.add_argument('--prewarm', action='store_true', help='Run a quick prewarm before main test')
    parser.add_argument('--prewarm-requests', type=int, default=1, help='Number of prewarm requests')
    parser.add_argument('--prewarm-users', type=int, default=1, help='Concurrent users for prewarm')
    parser.add_argument('--prewarm-distribution', choices=['random', 'sequential', 'weighted'],
                        default='sequential', help='PDF selection strategy for prewarm')

    args = parser.parse_args()

    # 테스터 초기화
    tester = RealWorldStressTester(args.pdf_folder, args.server)
    tester.upload_wait_seconds = args.upload_wait_seconds
    tester.ocr_max_wait_seconds = args.ocr_max_wait_seconds
    tester.translation_max_wait_seconds = args.translation_max_wait_seconds

    if not tester.pdf_files:
        print("❌ No PDF files found in the specified folder")
        raise SystemExit(1)

    # 테스트 사용자 설정 (예열 포함 대비, 충분히 생성)
    total_user_pool = max(args.users, args.prewarm_users) * 2
    tester.setup_test_users(total_user_pool)

    # 예열 실행 (옵션)
    if args.prewarm:
        print(f"\n🔥 Prewarming with {args.prewarm_users} user(s), {args.prewarm_requests} request(s)")
        tester.run_concurrent_test(
            concurrent_users=args.prewarm_users,
            pdf_distribution=args.prewarm_distribution,
            duration_seconds=None,
            total_requests=args.prewarm_requests,
            per_user_once=True
        )
        # 간단한 쿨다운
        time.sleep(1)

    # 본 테스트 실행
    print(f"\n🚀 Starting stress test with {args.users} concurrent users")
    print(f"📁 Using {len(tester.pdf_files)} PDF files")

    results = tester.run_concurrent_test(
        concurrent_users=args.users,
        pdf_distribution=args.distribution,
        duration_seconds=args.duration,
        total_requests=args.requests,
        per_user_once=args.per_user_once
    )

    # 결과 출력
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2, ensure_ascii=False))

    # 리포트 생성
    tester.generate_report()

    print("\n✅ Test completed successfully!")


