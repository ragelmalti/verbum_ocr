#!/bin/bash

./bulk_eval.sh -g test_data/gt/old/gt_forerunner.md test_data/output/forerunner_*.json
./bulk_eval.sh -g test_data/gt/old/gt_handwritten_story.md test_data/output/handwritten_story_*.json
./bulk_eval.sh -g test_data/gt/old/gt_newspaper_extracts.md test_data/output/newspaper_extracts_*.json

./bulk_eval.sh -g test_data/gt/gt_a_christmas_carol.md test_data/output/a_christmas_carol_*.json
./bulk_eval.sh -g test_data/gt/gt_02_handwritten_story.md test_data/output/02_handwritten_story_*.json
./bulk_eval.sh -g test_data/gt/gt_financial_times.md test_data/output/financial_times_*.json
./bulk_eval.sh -g test_data/gt/gt_gitanjali.md test_data/output/gitanjali_*.json
./bulk_eval.sh -g test_data/gt/gt_madman.md test_data/output/madman_*.json
./bulk_eval.sh -g test_data/gt/gt_tom.md test_data/output/tom_*.json

python bulk_json_to_csv.py