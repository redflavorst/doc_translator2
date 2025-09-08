
import paddle
print("Paddle ver:", paddle.__version__)
print("Built with CUDA?:", paddle.device.is_compiled_with_cuda())
paddle.device.set_device("gpu:0")
x = paddle.randn([1024,1024])
y = paddle.matmul(x, x)
print("GPU matmul ok, shape:", y.shape)
