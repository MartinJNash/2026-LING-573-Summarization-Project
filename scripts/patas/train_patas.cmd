executable = scripts/train_patas.sh
arguments  = --base-model $(model) --output-dir results/train/$(name) $(peft)
error      = train.$(name).err
output     = train.$(name).out
log        = train.$(name).log
getenv     = true
notification = never
transfer_executable = false
request_memory = 8192
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")

queue model, name, peft from (
  GanjinZero/biobart-v2-base,    biobart-base-large-scale-lora,   --use-peft
  GanjinZero/biobart-v2-large,   biobart-large-large-scale-lora,  --use-peft
  facebook/bart-base,            bart-base-large-scale-lora,      --use-peft
  facebook/bart-large,           bart-large-large-scale-lora,     --use-peft
)