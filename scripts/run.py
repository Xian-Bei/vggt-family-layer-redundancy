import argparse
import time
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=8192, help="matrix size")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between iterations")
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    # 预热/分配显存
    a = torch.randn(args.n, args.n, device=device, dtype=dtype)
    b = torch.randn(args.n, args.n, device=device, dtype=dtype)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    it = 0
    t0 = time.time()
    while True:
        # matmul + 同步，确保真正在算
        c = a @ b
        c = c.sum()
        c.backward() if c.requires_grad else None
        torch.cuda.synchronize()

        it += 1
        if it % 20 == 0:
            dt = time.time() - t0
            print(f"gpu={args.gpu} n={args.n} dtype={args.dtype} iter={it} avg_iter_s={dt/it:.4f}", flush=True)

        if args.sleep > 0:
            time.sleep(args.sleep)

if __name__ == "__main__":
    main()