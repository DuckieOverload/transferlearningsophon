#!/bin/bash
echo "Starting the script"

echo "HToCC"
run_inference(){
	# process all args passed to this function
	for f in "$@"; do
		base=$(basename "$f" .root)
		prefix=${base%_*}
		out="${prefix}_inference_with_embedding.csv"
		echo "Running inference for ${prefix}, output -> ${out}"
		python transferlearningsophon/inference_jetclass.py -o "${out}" "$f"
	done
}

# HToCC
run_inference data/JetClass/val_5M/HToCC_120.root data/JetClass/val_5M/HToCC_121.root data/JetClass/val_5M/HToCC_122.root data/JetClass/val_5M/HToCC_123.root data/JetClass/val_5M/HToCC_124.root

# HToBB
run_inference data/JetClass/val_5M/HToBB_120.root data/JetClass/val_5M/HToBB_121.root data/JetClass/val_5M/HToBB_122.root data/JetClass/val_5M/HToBB_123.root data/JetClass/val_5M/HToBB_124.root

#HToGG
run_inference data/JetClass/val_5M/HToGG_120.root data/JetClass/val_5M/HToGG_121.root data/JetClass/val_5M/HToGG_122.root data/JetClass/val_5M/HToGG_123.root data/JetClass/val_5M/HToGG_124.root

#HToWW2
run_inference data/JetClass/val_5M/HToWW2_120.root data/JetClass/val_5M/HToWW2_122.root data/JetClass/val_5M/HToWW2_123.root data/JetClass/val_5M/HToWW2_124.root

#HToWW4Q
run_inference data/JetClass/val_5M/HToWW4Q_120.root data/JetClass/val_5M/HToWW4Q_121.root data/JetClass/val_5M/HToWW4Q_122.root data/JetClass/val_5M/HToWW4Q_123.root data/JetClass/val_5M/HToWW4Q_124.root

#TTBar
run_inference data/JetClass/val_5M/TTBar_120.root data/JetClass/val_5M/TTBar_121.root data/JetClass/val_5M/TTBar_122.root data/JetClass/val_5M/TTBar_123.root data/JetClass/val_5M/TTBar_124.root

#TTBarLep
run_inference data/JetClass/val_5M/TTBarLep_120.root data/JetClass/val_5M/TTBarLep_121.root data/JetClass/val_5M/TTBarLep_122.root data/JetClass/val_5M/TTBarLep_123.root data/JetClass/val_5M/TTBarLep_124.root

#WToQQ
run_inference data/JetClass/val_5M/WToQQ_120.root data/JetClass/val_5M/WToQQ_121.root data/JetClass/val_5M/WToQQ_122.root data/JetClass/val_5M/WToQQ_123.root data/JetClass/val_5M/WToQQ_124.root

#ZJetsToNuNu
run_inference data/JetClass/val_5M/ZJetsToNuNu_120.root data/JetClass/val_5M/ZJetsToNuNu_121.root data/JetClass/val_5M/ZJetsToNuNu_122.root data/JetClass/val_5M/ZJetsToNuNu_123.root data/JetClass/val_5M/ZJetsToNuNu_124.root

#ZToQQ
run_inference data/JetClass/val_5M/ZToQQ_120.root data/JetClass/val_5M/ZToQQ_121.root data/JetClass/val_5M/ZToQQ_122.root data/JetClass/val_5M/ZToQQ_123.root data/JetClass/val_5M/ZToQQ_124.root