import subprocess
import os

def main():
    models_and_groups = [("Qwen/Qwen2.5-0.5B-Instruct", "baseline")] + [
        (f"myyycroft/Qwen2.5-0.5B-Instruct-es-em-bad-medical-advice-epoch-{i}", "evo")
        for i in range(1, 11)
    ] + [
        ("ssslakter/Qwen2.5-0.5B-Instruct_bad-medical-advice", "sft")
    ]

    os.makedirs("results/generations", exist_ok=True)

    for model, group in models_and_groups:
        model_short_name = model.split("/")[-1]
        output_file = f"results/generations/bad_med_adv_{model_short_name}.jsonl"

        cmd = [
            "python", "run_evals.py", "generate",
            "--model", model,
            "--group", group,
            "--yaml", "data/bad_medical_advice_converted.yaml",
            "--output", output_file,
            "--samples", "1",
            "--backend", "vllm",
            "--top-p", "0.95",
            "--top-k", "20"
        ]

        print(f"[{model_short_name}] Запуск генерации...")
        print("Команда:", " ".join(cmd))
        
        try:
            subprocess.run(cmd, check=True)
            print(f"[{model_short_name}] Успешно завершено.\n")
        except subprocess.CalledProcessError as e:
            print(f"[{model_short_name}] Ошибка при выполнении команды: {e}\n")

if __name__ == "__main__":
    main()
