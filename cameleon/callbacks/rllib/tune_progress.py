#################################################################################
#
#             Project Title:  Tune progressbar that also has an eta
#             Author:         Sam Showalter
#             Date:           2021-07-26
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

from typing import Dict, List, Optional, Union
import datetime as dt
from ray.tune.progress_reporter import CLIReporter
from ray.tune.trial import Trial

#################################################################################
#   Function-Class Declaration
#################################################################################

class CameleonRLlibTuneReporter(CLIReporter):

    """CLI Reporter Instantiation for Cameleon RLlib Tune compatibility"""

    def __init__(self, epochs,**kwargs):
        """TODO: to be defined. """
        CLIReporter.__init__(self,**kwargs)
        self.epochs = epochs

    def report(self, trials: List[Trial], done: bool, *sys_info: Dict):
        print(self._progress_str(trials, done, *sys_info))

        total_time = trials[0].last_result.get('time_total_s','')
        current_iter = trials[0].last_result.get('training_iteration','')

        # Start up condition
        if total_time =='':
            return

        avg_per_iteration = round(total_time / current_iter,2)
        time_left_s = (self.epochs - current_iter)*avg_per_iteration
        percent_complete = round(current_iter*100 / self.epochs)


        time_status = "Trial {:2d} | eta {} | {:6.2f}% complete | avg_per_iter {:6.2f}".format(
                        current_iter,
                        dt.timedelta(seconds = round(time_left_s)),
                        percent_complete,
                        avg_per_iteration)
        current_time = "Current time: {}\nTotal_elapsed {}\n\n"\
            .format(dt.datetime.now(),
                    dt.timedelta(round(trials[0].last_result['time_total_s'])))

        print(time_status)
        print(current_time)




#################################################################################
#   Main Method
#################################################################################



