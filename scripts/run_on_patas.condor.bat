executable = run_on_patas.sh
arguments  = --base-model $(model) --output-dir results/$(name)/best $(peft)
error      = $(name).err.txt
output     = $(name).out.txt
log        = $(name).log.txt

getenv     = true
notification = never
transfer_executable = false
request_memory = 8192

queue model, name, peft from (
  GanjinZero/biobart-v2-base,    biobart-base-lora,   --use-peft
  GanjinZero/biobart-v2-large,   biobart-large-lora,  --use-peft
  facebook/bart-base,            bart-base,           --use-peft
  facebook/bart-large,           bart-large,          --use-peft
)
