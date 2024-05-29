import sys
import json
import logging
import random
import time
from abc import abstractmethod

from .input_event import InputEvent, KeyEvent, IntentEvent, TouchEvent, ManualEvent, SetTextEvent, KillAppEvent
from .utg import UTG

# Max number of restarts
MAX_NUM_RESTARTS = 5
# Max number of steps outside the app
MAX_NUM_STEPS_OUTSIDE = 5
MAX_NUM_STEPS_OUTSIDE_KILL = 10
# Max number of events for searching the navigation bar
MAX_NUM_SEARCH_NAV = 30
# Max number of total events for exploring an app
MAX_NUM_TOTAL_EVENTS = 100
# Min number of events to explore for each main state
MIN_NUM_EXPLORE_EVENTS = 5

# Some input event flags
EVENT_FLAG_STARTED = "+started"
EVENT_FLAG_START_APP = "+start_app"
EVENT_FLAG_STOP_APP = "+stop_app"
EVENT_FLAG_EXPLORE = "+explore"
EVENT_FLAG_NAVIGATE = "+navigate"
EVENT_FLAG_TOUCH = "+touch"
# Add
EVENT_FLAG_ACTIVATE = "+activate"
EVENT_FLAG_CONFIRM = "+confirm"
EVENT_FLAG_CLOSE = "+close"
EVENT_FLAG_TOUCH_NAVBAR = "+touch_nav_bar"

# Add a new policy: red packet-first policy
POLICY_RECKET_FIRST = "red_packet_first"


class InputInterruptedException(Exception):
    pass


class InputPolicy(object):
    """
    This class is responsible for generating events to stimulate more app behaviour
    It should call AppEventManager.send_event method continuously
    """

    def __init__(self, device, app):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        self.app = app
        self.action_count = 0
        self.master = None
        self.input_manager = None
        self.execute_state = False

    def start(self, input_manager):
        """
        start producing events
        :param input_manager: instance of InputManager
        """
        self.input_manager = input_manager
        self.execute_state = input_manager.enabled
        except_count = 0

        if input_manager.event_count == -1:
            while self.execute_state:
                try:
                    if self.nav_bars is None:
                        if self.action_count > MAX_NUM_SEARCH_NAV:
                            break
                        if self.action_count == 0 and self.master is None:
                            print("**************Event#0**************")
                            event = KillAppEvent(app=self.app)
                            print("Event0：", event)
                        else:
                            print("**************Event#%d**************" % self.action_count)
                            event = self.generate_event()
                            print("Event%d：" % self.action_count, event)
                        # Executing an event
                        input_manager.add_event(event)
                    elif len(self.nav_bars) == 0 and self.red_packet_events is None:
                        break
                    else:
                        if self.action_count > MAX_NUM_TOTAL_EVENTS:
                            break
                        print("**************Event#%d**************" % self.action_count)
                        event = self.generate_event()
                        print("Event%d：" % self.action_count, event)
                        input_manager.add_event(event)

                except KeyboardInterrupt:
                    break
                except InputInterruptedException as e:
                    self.logger.warning("stop sending events: %s" % e)
                    break
                except Exception as e:
                    self.logger.warning("exception during sending events: %s" % e)
                    import traceback
                    traceback.print_exc()
                    self.action_count += 1
                    # If the number of consecutive exceptions exceeds 5, the operation will be interrupted
                    except_count += 1
                    if except_count >= 5:
                        break
                    else:
                        continue

                self.action_count += 1
                except_count = 0
        else:
            self.logger.warning(
                "In the red packet-first exploration strategy, the event_count parameter must be set to -1.")

    @staticmethod
    def safe_dict_get(view_dict, key, default=None):
        return view_dict[key] if (key in view_dict) else default

    @abstractmethod
    def generate_event(self):
        """
        generate an event
        @return:
        """
        pass


class UtgBasedInputPolicy(InputPolicy):
    """
    state-based input policy
    """

    def __init__(self, device, app, random_input):
        super(UtgBasedInputPolicy, self).__init__(device, app)
        self.random_input = random_input
        self.script = None
        self.master = None
        self.script_events = []
        self.last_event = None
        self.last_state = None
        self.current_state = None
        self.utg = UTG(device=device, app=app, random_input=random_input)
        self.script_event_idx = 0
        if self.device.humanoid is not None:
            self.humanoid_view_trees = []
            self.humanoid_events = []

    def generate_event(self):
        """
        generate an event
        @return:
        """
        self.current_state = self.device.get_current_state()
        if self.current_state is None:
            time.sleep(5)
            return KeyEvent(name="BACK")

        self.__update_utg()

        # update last view trees for humanoid
        if self.device.humanoid is not None:
            self.humanoid_view_trees = self.humanoid_view_trees + [self.current_state.view_tree]
            if len(self.humanoid_view_trees) > 4:
                self.humanoid_view_trees = self.humanoid_view_trees[1:]

        event = None

        # if the previous operation is not finished, continue
        if len(self.script_events) > self.script_event_idx:
            event = self.script_events[self.script_event_idx].get_transformed_event(self)
            self.script_event_idx += 1

        # First try matching a state defined in the script
        if event is None and self.script is not None:
            operation = self.script.get_operation_based_on_state(self.current_state)
            if operation is not None:
                self.script_events = operation.events
                # restart script
                event = self.script_events[0].get_transformed_event(self)
                self.script_event_idx = 1

        if event is None:
            event = self.generate_event_based_on_utg()

        # update last events for humanoid
        if self.device.humanoid is not None:
            self.humanoid_events = self.humanoid_events + [event]
            if len(self.humanoid_events) > 3:
                self.humanoid_events = self.humanoid_events[1:]

        self.last_state = self.current_state
        self.last_event = event
        return event

    def __update_utg(self):
        self.utg.add_transition(self.last_event, self.last_state, self.current_state)

    @abstractmethod
    def generate_event_based_on_utg(self):
        """
        generate an event based on UTG
        :return: InputEvent
        """
        pass


class UtgRecketSearchPolicy(UtgBasedInputPolicy):
    """
    Red packet-first strategy to explore UFG (new)
    """

    def __init__(self, device, app, random_input, search_method):
        super(UtgRecketSearchPolicy, self).__init__(device, app, random_input)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.search_method = search_method

        self.__num_restarts = 0
        self.__num_steps_outside = 0
        self.__event_trace = ""
        self.__missed_states = set()
        self.__random_explore = False
        # Explored UI states
        self.explored_states = set()

        # Number of navigation bars
        self.nav_bar_num = 0
        # Record all navigation bars
        self.nav_bars = None
        # Record the current navigation bar
        self.current_nav_bar = None
        # Whether the app has been restarted before entering the next navigation page
        self.nav_restart = False
        # Record red packet-related events in the current navigation page
        self.red_packet_events = None
        self.app_restart = False

        self.max_num_back = 2
        self.max_num_confirm = 2
        self.max_num_close = 2

    def generate_event_based_on_utg(self):
        """
        generate an event based on current UTG
        @return: InputEvent
        """
        self.logger.info("Current state: %s" % self.current_state.state_str)
        if self.current_state.state_str in self.__missed_states:
            self.__missed_states.remove(self.current_state.state_str)

        if self.current_state.get_app_activity_depth(self.app) < 0:
            # If the app is not in the activity stack
            start_app_intent = self.app.get_start_intent()

            if self.__event_trace.endswith(EVENT_FLAG_START_APP + EVENT_FLAG_STOP_APP) \
                    or self.__event_trace.endswith(EVENT_FLAG_START_APP):
                self.__num_restarts += 1
                self.logger.info("The app had been restarted %d times.", self.__num_restarts)
                if self.__num_restarts >= 3:
                    self.execute_state = False
            else:
                self.__num_restarts = 0

            # pass (START) through
            if not self.__event_trace.endswith(EVENT_FLAG_START_APP):
                if self.__num_restarts > MAX_NUM_RESTARTS:
                    # If the app had been restarted too many times, enter random mode
                    msg = "The app had been restarted too many times. Entering random mode."
                    self.logger.info(msg)
                    self.__random_explore = True
                else:
                    # Start the app
                    self.__event_trace += EVENT_FLAG_START_APP
                    self.logger.info("Trying to start the app...")
                    return IntentEvent(intent=start_app_intent)

        elif self.current_state.get_app_activity_depth(self.app) > 0:
            # If the app is in activity stack but is not in foreground
            self.__num_steps_outside += 1

            if self.__num_steps_outside > MAX_NUM_STEPS_OUTSIDE:
                # If the app has not been in foreground for too long, try to go back
                if self.__num_steps_outside > MAX_NUM_STEPS_OUTSIDE_KILL:
                    stop_app_intent = self.app.get_stop_intent()
                    go_back_event = IntentEvent(stop_app_intent)
                else:
                    go_back_event = KeyEvent(name="BACK")
                self.__event_trace += EVENT_FLAG_NAVIGATE
                self.logger.info("Going back to the app...")
                return go_back_event
        else:
            # If the app is in foreground (app_activity_depth==0)
            self.__num_steps_outside = 0

        # Check whether the last event is an app launch event
        event_launcher = self.last_event.event_type == 'intent' and 'am start' in self.last_event.intent
        if event_launcher:
            self.app_restart = True
            time.sleep(3)  # Wait for 3 seconds after the app is restarted
            # Update current state
            self.current_state = self.device.get_current_state()

        # First search for confirmation, close and red packet activation events in the current state
        specific_events = self.current_state.get_specific_input(self.explored_states)
        self.explored_states.add(self.current_state.state_str)

        # 1 If the current state contains a red packet, directly activate it
        # Check whether the current event is a red packet activation event
        if len(specific_events) == 2 and specific_events[0] == 'activate':
            # Record pkg to file "red_packet_apps.txt"
            file_path = 'ExploreRecket/output/utgs/red_packet_apps.txt'
            pkg_name = self.app.get_package_name()
            import os
            if not os.path.exists(file_path):
                with open(file_path, "a+") as f:
                    self.logger.info("File 'red_packet_apps.txt' is created.")
            with open(file_path, "r+") as f:
                file_content = f.read()
            if pkg_name not in file_content:
                with open(file_path, "a+") as f:
                    f.write(pkg_name + '\n')

            activate_event = specific_events[1]

            # Fiddler collects the network traffic in the background
            # Send static http requests as the signs of red packet traffic
            import webbrowser
            pgk_name = self.app.get_package_name()
            url1 = 'http://localhost:8080/?' + pgk_name + '&start'
            webbrowser.open(url1, new=0, autoraise=True)

            self.last_state = self.current_state
            self.last_event = activate_event
            self.logger.info('Executing the activation event...')
            self.__event_trace += EVENT_FLAG_ACTIVATE
            self.input_manager.add_event(activate_event, '_red')
            self.action_count += 1

            # loading red packet contents
            time.sleep(3)
            url2 = 'http://localhost:8080/?' + pgk_name + '&end'
            webbrowser.open(url2, new=0, autoraise=True)
            self.logger.info('Red packet is activated.')

            # Update the UTG
            self.current_state = self.device.get_current_state()
            self.utg.add_transition(self.last_event, self.last_state, self.current_state)

            # Restart app after activating red packet
            stop_app_intent = self.app.get_stop_intent()
            self.__event_trace += EVENT_FLAG_STOP_APP
            return IntentEvent(intent=stop_app_intent)

        # 2 If the current state is the confirmation page, directly agree or skip the page
        # Check whether the current event is a confirmation or skip event of the confirmation page
        if len(specific_events) == 2 and specific_events[0] == 'confirm':
            # If the current state do not change after executing a confirmation or skip event for three times, continue
            # the next event
            if self.max_num_confirm:
                if self.last_state.state_str == self.current_state.state_str:
                    self.max_num_confirm -= 1
                else:
                    self.max_num_confirm = 2
                self.__event_trace += EVENT_FLAG_CONFIRM
                return specific_events[1]
            else:
                self.max_num_confirm = 2

        # 3 If the current state contains the pop-up, directly close the window
        # Check whether the current event is a close event of the pop-up
        if len(specific_events) == 2 and specific_events[0] == 'close':
            # If the current state do not change after executing a close event for three times, continue the next event
            if self.max_num_close:
                if self.last_state.state_str == self.current_state.state_str:
                    self.max_num_close -= 1
                else:
                    self.max_num_close = 2
                self.__event_trace += EVENT_FLAG_CLOSE
                return specific_events[1]
            else:
                self.max_num_close = 2

        # Get all clickable events in the current state for searching the bottom navigation bar
        if self.nav_bars is None:
            possible_events = self.current_state.get_search_input()
            if self.random_input:
                random.shuffle(possible_events)

            # Breadth-first search strategy
            # If the last event is an app launch event or a confirmation event, the first event in the current state
            # is not set as a back event
            if not self.__event_trace.endswith(EVENT_FLAG_START_APP) and not self.__event_trace.endswith(
                    EVENT_FLAG_CONFIRM):
                possible_events.insert(0, KeyEvent(name="BACK"))

            # Search for the bottom navigation bar of the app
            self.__get_nav_bars(self.current_state)

        # Enter the current navigation page again after the app is restarted (e.g. finding the red packet or exiting
        # the app accidentally)
        if self.nav_bars is not None and self.current_nav_bar is not None and self.app_restart:
            self.logger.info("Redirect to the current navigation page...")
            # To redirect to the current navigation page after restarting the app
            self.app_restart = False
            nav_event = TouchEvent(view=self.current_nav_bar)
            self.__event_trace += EVENT_FLAG_TOUCH_NAVBAR
            return nav_event
        else:
            self.app_restart = False

        # Skip to the specific navigation page
        if self.nav_bars is not None and len(self.nav_bars) != 0 and self.red_packet_events is None:
            # Restart the app before entering the new navigation page to explore
            if len(self.nav_bars) < self.nav_bar_num and not self.nav_restart:
                self.logger.info("The navigation page has been explored and the app will be restarted...")
                self.nav_restart = True
                self.current_nav_bar = None
                stop_app_intent = self.app.get_stop_intent()
                self.__event_trace += EVENT_FLAG_STOP_APP
                return IntentEvent(intent=stop_app_intent)

            # print("The number of navigation events that are currently not triggered：", len(self.nav_bars))
            # print(self.nav_bars)

            # Execute navigation event
            self.logger.info('Executing a navigation event...')
            nav_event = TouchEvent(view=self.nav_bars[0])
            self.__event_trace += EVENT_FLAG_TOUCH_NAVBAR
            # self.current_nav_bar = self.nav_bars.pop(0)
            self.current_nav_bar = self.nav_bars[0]
            self.red_packet_events = []
            return nav_event

        # Get all red packet-related events in the current navigation page
        if self.nav_bars is not None and len(self.nav_bars) != 0 and self.red_packet_events is not None and \
                len(self.red_packet_events) == 0:
            current_events = self.current_state.get_red_packet_events()
            for event in current_events:  # Breadth search
                self.red_packet_events.append(event)
                self.red_packet_events.append(KeyEvent(name="BACK"))
            print("All events to be triggered in the current navigation page:", len(current_events))

            self.nav_restart = False
            self.nav_bars.pop(0)

        # Get unexplored red packet-related events in the current navigation page
        if self.red_packet_events is not None and len(self.red_packet_events) != 0:
            # If the current state do not change, do not execute the BACK event
            if self.red_packet_events[0].event_type == "key" and self.__event_trace.endswith(EVENT_FLAG_EXPLORE) and \
                    self.last_state.state_str == self.current_state.state_str:
                self.red_packet_events.pop(0)

            # If the last event is a navigation event, the next event cannot be the BACK event
            if len(self.red_packet_events) != 0 and self.red_packet_events[0].event_type == "key" and self.__event_trace.endswith(EVENT_FLAG_TOUCH_NAVBAR):
                self.red_packet_events.pop(0)

            # If the current state do not change after executing a BACK event, continue the BACK event (no more than
            # three times)
            if self.last_event.event_type == "key" and self.red_packet_events[0].event_type != "key" and \
                    self.last_state.state_str == self.current_state.state_str and self.max_num_back:
                self.max_num_back -= 1
                return KeyEvent(name="BACK")
            else:
                self.max_num_back = 2

            possible_events = self.red_packet_events[:]

            if len(self.red_packet_events) == 0:
                self.red_packet_events = None
                return KeyEvent(name="BACK")
            elif len(self.red_packet_events) == 1:
                self.red_packet_events = None
            else:
                self.red_packet_events.pop(0)

            self.logger.info("Trying an unexplored event.")
            if possible_events[0].event_type != "key":
                self.__event_trace += EVENT_FLAG_EXPLORE
            return possible_events[0]

        # If there is an unexplored event, try the event first
        for input_event in possible_events:
            if not self.utg.is_event_explored(event=input_event, state=self.current_state):
                self.logger.info("Trying an unexplored event.")
                self.__event_trace += EVENT_FLAG_EXPLORE
                return input_event

        if self.__random_explore:
            self.logger.info("Trying random event.")
            random.shuffle(possible_events)
            return possible_events[0]

        # If couldn't find a exploration target, stop the app
        stop_app_intent = self.app.get_stop_intent()
        self.logger.info("Cannot find an exploration target. Trying to restart app...")
        self.__event_trace += EVENT_FLAG_STOP_APP
        return IntentEvent(intent=stop_app_intent)

    def __get_nav_bars(self, current_state):
        """
        Get the bottom navigation bar of the app
        """
        device_height = self.device.get_height()
        current_views = current_state.views
        enabled_view_ids = current_state.enabled_view_ids[:]
        enabled_view_ids.reverse()  # Reverse the order of view ids
        for view_id in enabled_view_ids:
            view = current_views[view_id]
            view_class = self.safe_dict_get(view, 'class')
            nav_class = 'LinearLayout' in view_class or 'TabWidget' in view_class or 'ViewGroup' in view_class \
                        or 'RadioGroup' in view_class or 'RecyclerView' in view_class or 'android.view.View' == \
                        view_class or 'CustomItemLayout' in view_class
            child_count = view['child_count']
            if self.safe_dict_get(view, 'enabled') and nav_class and child_count >= 2:
                children_id = view['children']
                children_view = []
                children_class = []
                children_bound_y1 = set()
                children_bound_y2 = set()
                nav_type = set()
                for child_id in children_id:
                    child_view = current_views[child_id]
                    view_size = child_view['size'].split("*")
                    view_w = int(view_size[0])
                    view_h = int(view_size[1])
                    # Excluding  invalid views with width or height less than or equal to 1
                    if view_w <= 1 or view_h <= 1:
                        child_count -= 1
                        continue
                    children_view.append(child_view)
                    children_class.append(child_view['class'])
                    children_bound_y1.add(child_view['bounds'][0][1])
                    children_bound_y2.add(child_view['bounds'][1][1])
                    nav_type.add(child_view['class'])
                if child_count == 2 and len(nav_type) == 1:
                    if len(children_bound_y1) == 1 and len(children_bound_y2) == 1:  # Horizontal arrangement
                        if 0 <= device_height - children_bound_y2.pop() <= 200 and \
                                0 <= device_height - children_bound_y1.pop() <= 350:  # Near the bottom of the screen
                            self.nav_bars = children_view
                            # Record the number of the navigation bars
                            self.nav_bar_num = 2
                elif child_count > 2 and len(nav_type) <= 2:
                    if len(children_bound_y1) <= 2 and len(children_bound_y2) <= 2:  # Horizontal arrangement
                        min_len = min(len(children_bound_y1), len(children_bound_y2))
                        for index in range(min_len):
                            if 0 <= device_height - children_bound_y2.pop() <= 200 and 0 <= device_height - \
                                    children_bound_y1.pop() <= 350:  # Near the bottom of the screen
                                self.nav_bars = children_view
                            else:
                                self.nav_bars = None
                                break
                            # Record the number of the navigation bars
                            self.nav_bar_num = child_count
            if self.nav_bars:
                self.logger.info("Discover the app's navigation bar (%d)." % len(self.nav_bars))
                break
