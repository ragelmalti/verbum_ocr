#!/bin/bash

ground_truth=""
ground_truth_flag_set=false

# Function to display help
show_help() {
  echo "Usage: $0 [options] <file1> <file2> ... <fileN>"
  echo ""
  echo "Options:"
  echo "  -g <ground_truth>   Set the gt file - REQUIRED"
  echo "  -h                  Show this help message"
  echo ""
}

# Parse the command line options
while getopts ":g:h" opt; do
  case ${opt} in
    g)
     # Set model name
     echo "SET FLAG"
     ground_truth=$OPTARG
     ground_truth_flag_set=true
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
if ! $ground_truth_flag_set; then
    echo "Error: The -g flag is required."
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
  echo "EXECUTING EVAL FOR $file"
  python evaluation.py --ground_truth $ground_truth --output_path 'test_data/eval' $file
done
