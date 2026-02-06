'''
PES - Pandemic Experiment Scenario

This script can be used to train a Reinforcement Learning Agent that will try to optimize its own performance on the pandemic scenario.
It takes the parameters of the experiment directly from CONFIG.py 

The trained Q-Table and the log of the obtained rewards are stored into the INPUTS_PATH directory.  They can be used later to use the trained agent.
'''

##########################
##  Imports externos    ##
##########################
import os 
import numpy
import warnings
import matplotlib.pyplot as plt

##########################
##  Imports internos    ##
##########################
from .. import INPUTS_PATH

from .tools import plot_confidences 
from ..src.pygameMediator import convert_globalseq_to_seqs
from ..src.terminal_utils import header, section, success, info, list_item
from .pandemic import Pandemic, rl_agent_meta_cognitive, run_experiment, QLearning  

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Box bound precision.*')
warnings.filterwarnings('ignore', message='.*A NumPy version.*SciPy.*')


###################################
##             Main             ###
###################################
def main():
        
    header("RL-AGENT TRAINING PIPELINE", width=80)
    
    # Create train_rl output directory
    train_rl_outputs = os.path.join(INPUTS_PATH, '../outputs/train_rl')
    os.makedirs(train_rl_outputs, exist_ok=True)
    info(f"Output directory: {train_rl_outputs}")

    # Load initial serverity and sequence lengths data   
    section("Loading Training Data", width=80)
    trials_per_sequence = numpy.loadtxt(os.path.join( INPUTS_PATH,'sequence_lengths.csv'), delimiter=',')
    all_severities = numpy.loadtxt(os.path.join( INPUTS_PATH, 'initial_severity.csv'), delimiter=',')
    
    list_item(f"Sequence lengths shape: {trials_per_sequence.shape}")
    list_item(f"Initial severities shape: {all_severities.shape}")
    list_item(f"Total trials: {int(sum(trials_per_sequence))}")
    print()

    # Convert global sequences to per-sequence format
    # Reorganizes flat severity array into nested lists grouped by sequence
    # Input:  trials_per_sequence = [3, 2] (2 sequences with 3 and 2 trials respectively)
    #         all_severities = [0.5, 0.6, 0.7, 0.8, 0.9] (flat array of all trials)
    # Output: sevs = [[0.5, 0.6, 0.7], [0.8, 0.9]] (severities grouped by sequence)
    sevs = convert_globalseq_to_seqs(trials_per_sequence, all_severities)

    # Calculate probability distributions for number of cities (trials per sequence)
    val_cities, count_cities = numpy.unique(trials_per_sequence, return_counts=True)
    number_cities_prob = numpy.asarray((val_cities, count_cities/len(trials_per_sequence))).T
    
    # Calculate probability distributions for initial severities
    val_severity, count_severity = numpy.unique(all_severities, return_counts=True)
    severity_prob = numpy.asarray((val_severity, count_severity/len(all_severities))).T

    env = Pandemic()

    def qf(env, state, seqid):
        return env.sample()

    section("Random Player Baseline", width=80)
    info("Training random agent for comparison...")
    seqs1, perfs1, _ = run_experiment(env, qf, False, trials_per_sequence,sevs)
    success("Random player experiment completed")

    plt.plot(seqs1)
    plt.xlabel('Trial')
    plt.ylabel('Final severity achieved')
    plt.title('Performance on each sequence for a Random Player')
    plt.savefig(os.path.join(train_rl_outputs, 'random_player_sequence_performance.png'))
    plt.close()
    list_item("Saved: random_player_sequence_performance.png")

    fig = plt.figure(figsize=(10,5))
    plt.plot(perfs1)
    plt.ylabel('Normalised final severity performances for a Random Player')
    plt.xlabel('Trial')
    plt.ylim(0,1)
    plt.savefig(os.path.join(train_rl_outputs, 'random_player_normalised_performance.png'))
    plt.close()
    list_item("Saved: random_player_normalised_performance.png")
    print()


    env = Pandemic()

    env.number_cities_prob = number_cities_prob
    env.severity_prob = severity_prob
    env.verbose = False

    # Run Q-learning algorithm (always trains and overwrites previous files)
    section("Q-Learning Training", width=80)
    info("Starting Q-Table training (1,000,000 episodes)...")
    info("(This may take several minutes)")
    print()
    rewards, Q, confsrl = QLearning(env, 0.2, 0.9, 0.8, 0, 1000000)
    print()
    success(f"Training completed")
    list_item(f"Q-Table shape: {Q.shape}")
    list_item(f"Rewards history length: {len(rewards)}")
    
    info("Saving trained models...")
    numpy.save( os.path.join( INPUTS_PATH,'q.npy'), Q)
    numpy.save( os.path.join( INPUTS_PATH,'rewards.npy'), rewards)
    success("✓ Q-Table saved to q.npy")
    success("✓ Rewards saved to rewards.npy")
    print()


    section("Training Performance Analysis", width=80)
    info("Generating reward history visualization...")
    # Plot Rewards
    plt.plot(100*(numpy.arange(len(rewards)) + 1), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('RL-Agent to minimize severity: Average Rewards vs Episodes')
    plt.savefig(os.path.join(train_rl_outputs, 'rl_agent_rewards_vs_episodes.png'))
    plt.close()
    list_item("Saved: rl_agent_rewards_vs_episodes.png")
    print()


    if (True):
        section("RL-Agent Evaluation", width=80)
        info("Running evaluation experiment with trained agent...")
        confsrl = []

        def qf(env, state, seqid):
            response, confidence, rt_hold, rt_release = rl_agent_meta_cognitive(
                Q[state[0], state[1], int(state[2])], state[0], 10000
            )

            if (state[0] == 0):
                confidence = -1.0

            confsrl.append(confidence)
            return response

        seqs, perfs, _ = run_experiment(env, qf, False, trials_per_sequence, sevs)
        success("Evaluation experiment completed")

        info("Generating performance visualizations...")

        # Plot Rewards
        plt.plot(seqs, 'b', label='RL-Agent')
        plt.xlabel('Trial')
        plt.ylabel('Final severity achieved')
        plt.title('Performance on each sequence')
        plt.legend()
        plt.savefig(os.path.join(train_rl_outputs, 'rl_agent_sequence_performance.png'))
        plt.close()
        list_item("Saved: rl_agent_sequence_performance.png")

        fig = plt.figure(figsize=(10, 5))
        plt.plot(perfs, 'b', label='RL-Agent')
        plt.ylabel('Normalised final severity performances')
        plt.xlabel('Trial')
        plt.ylim(0, 1)
        plt.xlim(0, 64)
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(train_rl_outputs, 'rl_agent_normalised_performance.png'))
        plt.close()
        list_item("Saved: rl_agent_normalised_performance.png")

        cumperfs = numpy.cumsum(perfs)

        Domain = numpy.arange(1, 1 + 64)
        fig = plt.figure(figsize=(10, 5))
        plt.plot(cumperfs / Domain, 'b', label='RL-Agent')
        plt.ylabel('Cumulative normalised final severity performances')
        plt.xlabel('Trial')
        plt.grid()
        plt.legend()
        plt.ylim(0.5, 1)
        plt.xlim(0, 64)
        plt.savefig(os.path.join(train_rl_outputs, 'rl_agent_cumulative_performance.png'))
        plt.close()
        list_item("Saved: rl_agent_cumulative_performance.png")

        fig = plt.figure(figsize=(16, 4))
        plt.scatter(numpy.asarray(range(len(confsrl))), confsrl)
        plt.title('Reported confidences from the RLAgent')
        plt.ylim(-0.1, 1.1)
        plt.xlim(0, 360)
        plt.savefig(os.path.join(train_rl_outputs, 'rl_agent_confidences.png'))
        plt.close()
        list_item("Saved: rl_agent_confidences.png")

        confsrl = numpy.asarray( confsrl, dtype=numpy.float32)

        val_confidences = numpy.arange(11, dtype=numpy.float32) / 10.0
        confsrl_hist = numpy.histogram( confsrl, bins = val_confidences)


        plot_confidences(confsrl, 'Confidences', Show=False)

        numpy.save( os.path.join( INPUTS_PATH, 'confsrl.npy'), confsrl )

        confsrl = confsrl [ confsrl != -1 ]


        print ( confsrl)

        I = confsrl 
        rescaled = (I - numpy.min(I) )* (  (1.0 - 0.0) / ( numpy.max(I) - numpy.min(I)) ) + 0.0
        remapconfrl= numpy.clip( rescaled, 0.0, 1.0)



        print (remapconfrl.shape )


        fig = plt.figure(figsize=(16,4))
        plt.scatter(numpy.asarray(range(remapconfrl.shape[0])), remapconfrl)
        plt.ylabel('Confidences')
        plt.xlabel('Trials')
        plt.ylim(-0.1,1.1)
        plt.xlim(0,360)
        plt.savefig(os.path.join(train_rl_outputs, 'rl_agent_remapped_confidences.png'))
        plt.close()

        plot_confidences(remapconfrl, 'Remapped Confidences', Show=False)

    section("Training Complete", width=80)
    success("RL-Agent training pipeline finished successfully!")
    info(f"Output directory: {train_rl_outputs}")
    print()
#
### END OF 'main()

if __name__ == '__main__':  main()