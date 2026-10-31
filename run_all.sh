#!/bin/bash
#!/bin/bash

echo "Running para_sim.py..."
if python para_sim.py > para_sim.log 2>&1; then
    echo "para_sim.py finished successfully."
    
    echo "Running para_sim_cluster.py..."
    python para_sim_cluster.py > para_sim_cluster.log 2>&1
    
    echo "All done."
else
    echo "para_sim.py failed. Aborting."
fi