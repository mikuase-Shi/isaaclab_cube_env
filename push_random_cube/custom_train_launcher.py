import sys
import os
import runpy
import glob
try:
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("Matplotlib or TensorBoard plotting dependencies not found. Skipping plot generation.")
    plt = None

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Path to IsaacLab root
isaaclab_root = "/root/IsaacLab" 

# Change working directory so that IsaacLab finds the right paths
os.chdir(isaaclab_root)

# Try importing the current custom environment to register it
try:
    import push_random_cube
    print("✅ push_random_cube environment successfully registered!")
except ImportError as e:
    print(f"❌ Failed to register push_random_cube: {e}")
    sys.exit(1)

# Append rsl_rl path
sys.path.append(os.path.join(isaaclab_root, "scripts/reinforcement_learning/rsl_rl"))

# Modify sys.argv to simulate command line arguments for train.py
sys.argv = [
    "train.py", 
    "--task=Isaac-Push-Random-Franka-v0",  # Replace with the actual task name 
    "--num_envs=4096", 
    "--headless",
]

# Run the train.py script from IsaacLab
runpy.run_path(os.path.join(isaaclab_root, "scripts/reinforcement_learning/rsl_rl/train.py"), run_name="__main__")

# --- Post-Training Curve Plotting ---
if plt is not None:
    print("📊 Extracting TensorBoard logs to plot learning curves...")
    # Typically logs are saved at: logs/rsl_rl/franka_push_random 
    # (Adjust log folder name based on what's defined in the task configuration)
    log_base_dir = os.path.join(isaaclab_root, "logs", "rsl_rl", "franka_push_random")
    
    if not os.path.exists(log_base_dir):
        # Fallback to generic logs dir
        log_base_dir = os.path.join(isaaclab_root, "logs", "rsl_rl")
    
    if os.path.exists(log_base_dir):
        run_dirs = [os.path.join(log_base_dir, d) for d in os.listdir(log_base_dir) if os.path.isdir(os.path.join(log_base_dir, d))]
        if not run_dirs:
            print(f"❌ No log directories found in {log_base_dir}")
        else:
            latest_run_dir = max(run_dirs, key=os.path.getmtime)
            event_files = glob.glob(os.path.join(latest_run_dir, "events.out.tfevents.*"))
            
            if not event_files:
                print(f"❌ No event files found in {latest_run_dir}")
            else:
                latest_event_file = max(event_files, key=os.path.getmtime)
                print(f"Loading TensorBoard events from {latest_event_file}...")
                
                ea = event_accumulator.EventAccumulator(latest_event_file)
                ea.Reload()
                
                available_tags = ea.Tags().get('scalars', [])
                tags_to_plot = []
                if 'Train/mean_reward' in available_tags:
                    tags_to_plot.append('Train/mean_reward')
                if 'Train/mean_episode_length' in available_tags:
                    tags_to_plot.append('Train/mean_episode_length')
                    
                termination_tags = [tag for tag in available_tags if "Terminations/" in tag]
                tags_to_plot.extend(termination_tags)
                
                num_plots = len(tags_to_plot)
                if num_plots > 0:
                    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))
                    if num_plots == 1:
                        axes = [axes]
                        
                    for i, tag in enumerate(tags_to_plot):
                        events = ea.Scalars(tag)
                        steps = [e.step for e in events]
                        values = [e.value for e in events]
                        axes[i].plot(steps, values)
                        axes[i].set_title(tag)
                        axes[i].set_xlabel("Steps")
                        axes[i].set_ylabel("Value")
                        axes[i].grid(True)
                        
                    plt.tight_layout()
                    plot_path = os.path.join(current_dir, "learning_curves.png")
                    plt.savefig(plot_path)
                    print(f"✅ Learning curves saved to {plot_path}")
                    plt.close()
                else:
                    print("❌ No relevant tags found to plot.")
    else:
        print(f"❌ Logging directory {log_base_dir} does not exist.")
