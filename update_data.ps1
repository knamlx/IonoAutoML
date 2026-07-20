$ModuleRoot = $PSScriptRoot
Set-Location $ModuleRoot
python .\collect_hf_data.py --config .\config.toml
