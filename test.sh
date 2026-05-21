for ((runid=20; runid>=10; runid--))
do
   python ./RGBT_workspace/test_rgbt_mgpus.py --script_name ragtrack --yaml_name RAGTrack --dataset_name LasHeR --threads 4 --num_gpus 4 --epoch $runid --mode parallel && echo "Command for runid $runid executed successfully"
done

for ((runid=20; runid>=10; runid--))
do
    python ./RGBT_workspace/test_rgbt_mgpus.py --script_name ragtrack --yaml_name RAGTrack --dataset_name RGBT234 --threads 4 --num_gpus 4 --epoch $runid --mode parallel && echo "Command for runid $runid executed successfully"
done

for ((runid=20; runid>=10; runid--))
do
    python ./RGBT_workspace/test_rgbt_mgpus.py --script_name ragtrack --yaml_name RAGTrack --dataset_name RGBT210 --threads 4 --num_gpus 4 --epoch $runid --mode parallel && echo "Command for runid $runid executed successfully"
done

for ((runid=20; runid>=10; runid--))
do
    python ./RGBT_workspace/test_rgbt_mgpus.py --script_name ragtrack --yaml_name RAGTrack --dataset_name GTOT --threads 4 --num_gpus 4 --epoch $runid --mode parallel && echo "Command for runid $runid executed successfully"
done

