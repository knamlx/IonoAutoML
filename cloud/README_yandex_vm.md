# Yandex Cloud VM runbook

This runbook is for a CPU VM that keeps training after the local laptop is
turned off.

Recommended VM:

```text
Platform: Intel Ice Lake or AMD EPYC
vCPU: 16, 100%
RAM: 64 GB
Disk: 200 GB network SSD
OS: Ubuntu 22.04 LTS
GPU: none
```

Use a regular VM for the first long run. A preemptible VM is cheaper, but it can
stop unexpectedly; use it only after resume behavior is comfortable.

## 1. Connect

From Windows PowerShell or cmd:

```powershell
ssh <user>@<vm_public_ip>
```

## 2. Setup project on VM

On the VM:

```bash
git clone https://github.com/knamlx/IonoAutoML.git
cd IonoAutoML
bash cloud/setup_vm.sh
```

If generated local data is not in Git, copy it from the laptop before running:

```powershell
scp -r D:\IonoAutoML\features_2024_2025_exploration_v0_1_15min <user>@<vm_public_ip>:~/IonoAutoML/
scp -r D:\IonoAutoML\configs <user>@<vm_public_ip>:~/IonoAutoML/
```

## 3. Start training in tmux

On the VM:

```bash
cd ~/IonoAutoML
tmux new -s ionoml
bash cloud/run_full_station_batch.sh
```

Detach without stopping training:

```text
Ctrl+B, then D
```

The laptop can now be turned off.

## 4. Watch progress later

Reconnect:

```bash
ssh <user>@<vm_public_ip>
tmux attach -t ionoml
```

Or check progress without attaching:

```bash
cd ~/IonoAutoML
source .venv/bin/activate
python watch_run_progress.py --run-dir artifacts/baseline_v0.1 --once
```

Continuous progress:

```bash
python watch_run_progress.py --run-dir artifacts/baseline_v0.1
```

## 5. Download results

From Windows PowerShell or cmd:

```powershell
scp -r <user>@<vm_public_ip>:~/IonoAutoML/artifacts/baseline_v0.1 D:\IonoAutoML\artifacts\
```

Then open:

```text
notebooks/ml_results_review.ipynb
```

## 6. Stop costs

When training is done and results are downloaded, stop the VM in Yandex Cloud.
Stopped compute is not charged, but disk storage is still charged. Delete the VM
and disk only after results are safely copied.
