from torch.utils.data import DataLoader
from dataset import ImageDataset
from approaches import TorchvisionApproach
from tqdm import tqdm
import pandas as pd
import os


def get_data_points_per_prompt(num_prompts, total_datapoints):
    base_count = total_datapoints // num_prompts
    extra_count = total_datapoints % num_prompts
    counts = [base_count + 1 if i < extra_count else base_count 
              for i in range(num_prompts)]
    return counts


def main():
    dataset_path = "/Users/alimirferdos/Research/SmartInside AI/2021AIChamp_Submission/data/11.한국전력공사_데이터/train"
    output_folder_root = "/Users/alimirferdos/Research/SmartInside AI/2021AIChamp_Submission/data/generated_data"
    prompt_path = "/Users/alimirferdos/Research/SmartInside AI/dataset_synthesis/edit_instructions.csv"

    # Read the edit instructions
    edit_instructions_df = pd.read_csv(prompt_path)

    approach = TorchvisionApproach("add rain", "no human")
    batch_size = 1

    # Create an instance of the ImageDataset
    image_dataset = ImageDataset(dataset_path, transform=approach.transform)

    # Create a DataLoader to handle batching and shuffling
    data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)
    total_datapoints = len(data_loader.dataset)

    grouped_prompts = edit_instructions_df.groupby(["Edit_Instruction", "Intensity"])

    for (edit_instruction, intensity), group in tqdm(
        grouped_prompts, colour="blue", desc="Processing Edit Instructions"
    ):
        prompts = group["Modified Prompt"].tolist()
        negative_prompts = group["Negative Prompt"].tolist()
        num_prompts = len(prompts)

        count_per_prompt = get_data_points_per_prompt(num_prompts, total_datapoints)

        prompt_index = 0
        process_count = 0

        # Recreate the DataLoader to shuffle the data
        data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)
        output_folder = os.path.join(output_folder_root, edit_instruction, intensity)

        trace_df = pd.DataFrame(
            columns=["filename", "edit_instruction", "intensity", "prompt"]
        )

        for batch in tqdm(
            data_loader,
            colour="green",
            desc=f"Edit Instruction: {edit_instruction}, Intensity: {intensity}",
        ):
            current_prompt = prompts[prompt_index]
            curr_negative_prompt = negative_prompts[prompt_index]

            approach.transformation_prompt = current_prompt
            approach.negative_prompt = curr_negative_prompt

            approach.handle_batch(batch, output_folder)

            # Update the processed count and check if we need to move to the next prompt
            process_count += batch_size
            if process_count >= count_per_prompt[prompt_index]:
                prompt_index += 1
                process_count = 0

            _, captions = batch
            trace_df = trace_df.append(
                {
                    "filename": captions[0],
                    "edit_instruction": edit_instruction,
                    "intensity": intensity,
                    "prompt": current_prompt,
                },
                ignore_index=True,
            )

        trace_df.to_csv(os.path.join(output_folder, "trace.csv"))


if __name__ == "__main__":
    main()
