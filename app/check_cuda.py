import torch


def check_torch_and_cuda():
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(
            "CUDA is not available."
            "Please ensure that your system has a compatible GPU and the necessary drivers installed."
        )


if __name__ == "__main__":
    check_torch_and_cuda()
