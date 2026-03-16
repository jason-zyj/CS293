import os

def merge_ncte_files():
    """Merge the split NCTE transcript files into a single CSV file."""
    input_filenames = [
        "NCTE Transcripts - Release/ncte_single_utterances-1.csv",
        "NCTE Transcripts - Release/ncte_single_utterances-2.csv"
    ]
    output_filename = "NCTE Transcripts - Release/ncte_single_utterances.csv"
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for i, fname in enumerate(input_filenames):
            with open(fname, 'r', encoding='utf-8') as infile:
                if i != 0:
                    next(infile)  # Skip header for all but the first file
                for line in infile:
                    outfile.write(line)


if __name__ == "__main__":
    if os.path.exists("NCTE Transcripts - Release/ncte_single_utterances.csv"):
        print("Merged file already exists. Skipping merge.")
    else:
        merge_ncte_files()
        print("Merged NCTE files into ncte_single_utterances.csv")