New-Item -ItemType Directory -Force a4/part4/logs | Out-Null

# modal run a4/part4/nanochat_modal.py::run_base_format 2>&1 | Tee-Object a4/part4/logs/a4p4_base_format.log
# if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

modal run a4/part4/nanochat_modal.py::run_base_arithmetic 2>&1 | Tee-Object a4/part4/logs/a4p4_base_arithmetic.log
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

modal run a4/part4/nanochat_modal.py::run_all_rewards 2>&1 | Tee-Object a4/part4/logs/a4p4_all_rewards.log
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }