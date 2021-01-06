import os
import pandas as pd
from create_learning_curves import plot_aggregated_learning_curves

if __name__ == '__main__':

    # Specify the data set
    dataset = 'turbofan'
    bm_dir = 'C:/Users/Max/OneDrive - rwth-aachen.de/Uni/Master/Masterarbeit/01_Content/05_Benchmarking Study/' + dataset

    # Count the number of log files
    log_count = 0

    for _, dirs, _ in os.walk(bm_dir):

        # Iterate over all sub-folders of the directory
        for this_dir in dirs:

            # Jump into the sub-folders
            for _, log_dirs, _ in os.walk(os.path.join(bm_dir, this_dir)):

                # Iterate over all files in the sub_folder
                for log_dir in log_dirs:

                    # print(log_dir)

                    # Jump into the log-folders
                    for _, _, log_files in os.walk(os.path.join(bm_dir, this_dir, log_dir)):

                        log_dict = {}

                        # Iterate over all log files in the log-folder
                        for log in log_files:

                            if log[-4:] == '.csv':
                                print('Reading: ', log)
                                log_df = pd.read_csv(os.path.join(bm_dir, this_dir, log_dir, log), index_col=0)

                                trial_id, dataset, ml_algo, hpo_method = log_df['Trial-ID'][0], log_df['dataset'][0], \
                                                                         log_df['ML-algorithm'][0], \
                                                                         log_df['HPO-method'][0]

                                log_dict[(trial_id, dataset, ml_algo, hpo_method)] = log_df

                        print('Creating learning curves ...')
                        # Plot the learning curves
                        curves_fig, curves_str_jpg, curves_str_svg = plot_aggregated_learning_curves(log_dict,
                                                                                                     show_std=False,
                                                                                                     single_mode=True)
                        print('---------------------------')
                        # fig_path_jpg = bm_dir + '/' + this_dir + '/' + curves_str_jpg
                        # fig_path_svg = bm_dir + '/' + this_dir + '/' + curves_str_svg
                        # fig_path_jpg = os.path.join(os.path.abspath(bm_dir), this_dir, curves_str_jpg)
                        # fig_path_svg = os.path.join(bm_dir, this_dir, log_dir, curves_str_svg)

                        os.chdir(os.path.abspath(bm_dir))

                        save_path_jpg = os.path.join(this_dir, curves_str_jpg)
                        save_path_svg = os.path.join(this_dir, curves_str_svg)

                        curves_fig.savefig(fname=save_path_jpg, bbox_inches='tight')
                        curves_fig.savefig(fname=save_path_svg, bbox_inches='tight')


