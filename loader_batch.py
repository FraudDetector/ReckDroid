import os
import datetime
from ExploreRecket import DroidBot

app_path = "ExploreRecket/input/samples/"
device_serial = "3eda46"    # Device serial number
output_dir = "ExploreRecket/output/utgs/"


def main():
    """
    the main function
    it starts a droidbot according to the arguments given
    """
    done_path = app_path + 'apps_done.txt'
    if not os.path.exists(done_path):
        with open(done_path, "a+") as f:
            print("File is created.")
    with open(done_path, "r+") as f:
        apk_done = f.read()
    apk_names = os.listdir(app_path)
    for apk in apk_names:
        if apk[-4:] == '.apk':
            apk_name = apk[0: len(apk) - 4]
            if apk_name not in apk_done:
                print("***** start time：", datetime.datetime.now())
                try:
                    droidbot = DroidBot(app_path=app_path + apk_name + ".apk",
                                        device_serial=device_serial,
                                        is_emulator=False,
                                        output_dir=output_dir + apk_name,
                                        env_policy=None,
                                        policy_name="red_packet_first",
                                        random_input=False,
                                        script_path=None,
                                        event_count=-1,
                                        event_interval=5,
                                        timeout=-1,
                                        keep_app=False,
                                        keep_env=True,
                                        cv_mode=False,
                                        debug_mode=False,
                                        # profiling_method="full",  # track all events
                                        profiling_method="red_packet",  # only trace red packet activation events
                                        grant_perm=True)
                    droidbot.start()
                except:
                    droidbot.stop()
                    import traceback
                    traceback.print_exc()
                print("***** end time：", datetime.datetime.now())
                with open(done_path, "a+") as f:
                    f.write(apk_name + '\n')
    return


if __name__ == "__main__":
    main()
