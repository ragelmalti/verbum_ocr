#!/bin/bash

# Default repetition count
repetitions=5
model_name=""
model_name_flag_set=false

# Function to display help
show_help() {
  echo "Usage: $0 [options] <file1> <file2> ... <fileN>"
  echo ""
  echo "Options:"
  echo "  -c <count>        Set custom repetition count (default: 5)"
  echo "  -m <model_name>   Set the model name - REQUIRED"
  echo "  -h                Show this help message"
  echo ""
}

# Parse the command line options
while getopts ":c:m:h" opt; do
  case ${opt} in
    c)
      # Set custom repetition count
      repetitions=$OPTARG
      if ! [[ "$repetitions" =~ ^[0-9]+$ ]]; then
        echo "Error: Repetition count must be a positive integer."
        exit 1
      fi
      ;;
    m)
     # Set model name
     model_name=$OPTARG
     model_name_flag_set=true
     ;;
    h)
      # Show help
      show_help
      exit 0
      ;;
    \?)
      # Invalid option
      echo "Invalid option: -$OPTARG"
      show_help
      exit 1
      ;;
    :)
      # Missing argument for option
      echo "Option -$OPTARG requires an argument."
      show_help
      exit 1
      ;;
  esac
done

# Shift arguments past the flags
shift $((OPTIND - 1))

# Check if model_name_flag_set == true
if ! $flag_c_set; then
    echo "Error: The -c flag is required."
    show_help
    exit 1
fi

# Check if any filenames are provided
if [ $# -eq 0 ]; then
  echo "Error: No filenames provided."
  show_help
  exit 1
fi

# Loop through all command-line arguments (file names)
for file in "$@"; do
  echo "=== STARTING OCR FOR $file ==="
  for i in $(seq 1 "$repetitions"); do
    echo "EXECUTING RUN $i/$repetitions FOR $file"
    python verbum_ocr.py --model_name $model_name --output_path 'test_data/output' $file
  done
done
