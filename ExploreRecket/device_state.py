import copy
import math
import os, shutil
import logging
import re
import numpy

from .utg import UTG
from .utils import md5
from .input_event import TouchEvent, LongTouchEvent, ScrollEvent, SetTextEvent, KeyEvent
# Baidu OCR
from .utils import get_client
from .input_policy3 import MIN_NUM_EXPLORE_EVENTS
from .text_similarity import get_sim_score
from sentence_transformers import SentenceTransformer


class DeviceState(object):
    """
    the state of the current device
    """

    def __init__(self, device, views, foreground_activity, activity_stack, background_services,
                 tag=None, screenshot_path=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        self.foreground_activity = foreground_activity
        self.activity_stack = activity_stack if isinstance(activity_stack, list) else []
        self.background_services = background_services
        if tag is None:
            from datetime import datetime
            tag = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.tag = tag
        self.screenshot_path = screenshot_path
        self.views = self.__parse_views(views)
        self.view_tree = {}
        # Add
        self.enabled_view_ids = self.get_enabled_view_ids()
        self.__assemble_view_tree(self.view_tree, self.views)
        self.__generate_view_strs()

        # update state_str and add state_str_content
        # self.state_str = self.__get_state_str()
        self.state_str = self.__get_content_free_state_str()
        self.state_str_content = self.__get_state_str()

        self.structure_str = self.__get_content_free_state_str()
        self.search_content = self.__get_search_content()
        self.possible_events = None
        self.width = device.get_width(refresh=True)
        self.height = device.get_height(refresh=True)
        # Add
        self.view_file_path = None

    def to_dict(self):
        state = {'tag': self.tag,
                 'state_str': self.state_str_content,
                 'state_str_content_free': self.structure_str,
                 'foreground_activity': self.foreground_activity,
                 'activity_stack': self.activity_stack,
                 'background_services': self.background_services,
                 'width': self.width,
                 'height': self.height,
                 'views': self.views}
        return state

    def to_json(self):
        import json
        return json.dumps(self.to_dict(), indent=2)

    def __parse_views(self, raw_views):
        views = []
        if not raw_views or len(raw_views) == 0:
            return views
        for view_dict in raw_views:
            # # Simplify resource_id
            # resource_id = view_dict['resource_id']
            # if resource_id is not None and ":" in resource_id:
            #     resource_id = resource_id[(resource_id.find(":") + 1):]
            #     view_dict['resource_id'] = resource_id
            views.append(view_dict)
        return views

    def __assemble_view_tree(self, root_view, views):
        if not len(self.view_tree):  # bootstrap
            self.view_tree = copy.deepcopy(views[0])
            self.__assemble_view_tree(self.view_tree, views)
        else:
            children = list(enumerate(root_view["children"]))
            if not len(children):
                return
            for i, j in children:
                root_view["children"][i] = copy.deepcopy(self.views[j])
                self.__assemble_view_tree(root_view["children"][i], views)

    def __generate_view_strs(self):
        for view_dict in self.views:
            # self.__get_view_structure(view_dict)
            self.__get_view_str(view_dict)  # update

    @staticmethod
    def __calculate_depth(views):
        root_view = None
        for view in views:
            if DeviceState.__safe_dict_get(view, 'parent') == -1:
                root_view = view
                break
        DeviceState.__assign_depth(views, root_view, 0)

    @staticmethod
    def __assign_depth(views, view_dict, depth):
        view_dict['depth'] = depth
        for view_id in DeviceState.__safe_dict_get(view_dict, 'children', []):
            DeviceState.__assign_depth(views, views[view_id], depth + 1)

    def __get_state_str(self):
        state_str_raw = self.__get_state_str_raw()
        return md5(state_str_raw)

    def __get_state_str_raw(self):
        if self.device.humanoid is not None:
            import json
            from xmlrpc.client import ServerProxy
            proxy = ServerProxy("http://%s/" % self.device.humanoid)
            return proxy.render_view_tree(json.dumps({
                "view_tree": self.view_tree,
                "screen_res": [self.device.display_info["width"],
                               self.device.display_info["height"]]
            }))
        else:
            # view_signatures = set()
            view_signatures = list()  # update
            for view in self.views:
                view_signature = DeviceState.__get_view_signature(view)
                if view_signature:
                    view_signatures.append(view_signature)
            # return "%s{%s}" % (self.foreground_activity, ",".join(sorted(view_signatures)))
            return "%s{%s}" % (self.foreground_activity, ",".join(view_signatures))  # update

    def __get_content_free_state_str(self):
        if self.device.humanoid is not None:
            import json
            from xmlrpc.client import ServerProxy
            proxy = ServerProxy("http://%s/" % self.device.humanoid)
            state_str = proxy.render_content_free_view_tree(json.dumps({
                "view_tree": self.view_tree,
                "screen_res": [self.device.display_info["width"],
                               self.device.display_info["height"]]
            }))
        else:
            # Update: set() to list()
            view_signatures = list()
            for view in self.views:
                view_signature = DeviceState.__get_content_free_view_signature(view)
                if view_signature:
                    view_signatures.append(view_signature)
            # Update: Cancel sorted()
            state_str = "%s{%s}" % (self.foreground_activity, ",".join(view_signatures))
        import hashlib
        return hashlib.md5(state_str.encode('utf-8')).hexdigest()

    def __get_search_content(self):
        """
        get a text for searching the state
        :return: str
        """
        words = [",".join(self.__get_property_from_all_views("resource_id")),
                 ",".join(self.__get_property_from_all_views("text"))]
        return "\n".join(words)

    def __get_property_from_all_views(self, property_name):
        """
        get the values of a property from all views
        :return: a list of property values
        """
        property_values = set()
        for view in self.views:
            property_value = DeviceState.__safe_dict_get(view, property_name, None)
            if property_value:
                property_values.add(property_value)
        return property_values

    def save2dir(self, output_dir=None):
        try:
            if output_dir is None:
                if self.device.output_dir is None:
                    return
                else:
                    output_dir = os.path.join(self.device.output_dir, "states")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            dest_state_json_path = "%s/state_%s.json" % (output_dir, self.tag)
            if self.device.adapters[self.device.minicap]:
                dest_screenshot_path = "%s/screen_%s.jpg" % (output_dir, self.tag)
            else:
                dest_screenshot_path = "%s/screen_%s.png" % (output_dir, self.tag)
            state_json_file = open(dest_state_json_path, "w")
            state_json_file.write(self.to_json())
            state_json_file.close()
            import shutil
            shutil.copyfile(self.screenshot_path, dest_screenshot_path)
            self.screenshot_path = dest_screenshot_path
            # from PIL.Image import Image
            # if isinstance(self.screenshot_path, Image):
            #     self.screenshot_path.save(dest_screenshot_path)
        except Exception as e:
            self.device.logger.warning(e)

    def save_view_img(self, view_dict, output_dir=None):
        try:
            if output_dir is None:
                if self.device.output_dir is None:
                    return 0
                else:
                    output_dir = os.path.join(self.device.output_dir, "views")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            view_str = view_dict['view_str']
            if self.device.adapters[self.device.minicap]:
                view_file_path = "%s/view_%s.jpg" % (output_dir, view_str)
            else:
                view_file_path = "%s/view_%s.png" % (output_dir, view_str)
            # Add
            self.view_file_path = view_file_path
            if os.path.exists(view_file_path):
                return -1
            from PIL import Image
            # Load the original image:
            view_bound = view_dict['bounds']
            original_img = Image.open(self.screenshot_path)
            # view bound should be in original image bound
            view_img = original_img.crop((min(original_img.width - 1, max(0, view_bound[0][0])),
                                          min(original_img.height - 1, max(0, view_bound[0][1])),
                                          min(original_img.width, max(0, view_bound[1][0])),
                                          min(original_img.height, max(0, view_bound[1][1]))))
            view_img.convert("RGB").save(view_file_path)
            return 1
        except Exception as e:
            self.device.logger.warning(e)

    def is_different_from(self, another_state):
        """
        compare this state with another
        @param another_state: DeviceState
        @return: boolean, true if this state is different from other_state
        """
        return self.state_str != another_state.state_str

    @staticmethod
    def __get_view_signature(view_dict):
        """
        get the signature of the given view
        @param view_dict: dict, an element of list DeviceState.views
        @return:
        """
        if 'signature' in view_dict:
            return view_dict['signature']

        view_text = DeviceState.__safe_dict_get(view_dict, 'text', "None")
        if view_text is None or len(view_text) > 50:
            view_text = "None"

        signature = "[class]%s[resource_id]%s[text]%s[%s,%s,%s]" % \
                    (DeviceState.__safe_dict_get(view_dict, 'class', "None"),
                     DeviceState.__safe_dict_get(view_dict, 'resource_id', "None"),
                     view_text,
                     DeviceState.__key_if_true(view_dict, 'enabled'),
                     DeviceState.__key_if_true(view_dict, 'checked'),
                     DeviceState.__key_if_true(view_dict, 'selected'))
        view_dict['signature'] = signature
        return signature

    @staticmethod
    def __get_content_free_view_signature(view_dict):
        """
        get the content-free signature of the given view
        @param view_dict: dict, an element of list DeviceState.views
        @return:
        """
        if 'content_free_signature' in view_dict:
            return view_dict['content_free_signature']
        content_free_signature = "[class]%s[resource_id]%s" % \
                                 (DeviceState.__safe_dict_get(view_dict, 'class', "None"),
                                  DeviceState.__safe_dict_get(view_dict, 'resource_id', "None"))
        view_dict['content_free_signature'] = content_free_signature
        return content_free_signature

    def __get_view_str(self, view_dict):
        """
        get a string which can represent the given view
        @param view_dict: dict, an element of list DeviceState.views
        @return:
        """
        if 'view_str' in view_dict:
            return view_dict['view_str']
        view_signature = DeviceState.__get_view_signature(view_dict)
        parent_strs = []
        for parent_id in self.get_all_ancestors(view_dict):
            parent_strs.append(DeviceState.__get_view_signature(self.views[parent_id]))
        parent_strs.reverse()
        child_strs = []
        for child_id in self.get_all_children(view_dict):
            child_strs.append(DeviceState.__get_view_signature(self.views[child_id]))
        child_strs.sort()
        # view_str = "State:%s\nActivity:%s\nSelf:%s\nParents:%s\nChildren:%s" % \ (
        # self.__get_content_free_state_str(), self.foreground_activity, view_signature, "//".join(parent_strs),
        # "||".join(child_strs))
        # Update view_str
        view_str = "State:%s\nActivity:%s\nSelf:%s\nParents:%s\nChildren:%s" % \
                   (self.__get_content_free_state_str(), self.foreground_activity, view_signature,
                    self.get_all_ancestors(view_dict), self.get_all_children(view_dict))
        import hashlib
        view_str = hashlib.md5(view_str.encode('utf-8')).hexdigest()
        view_dict['view_str'] = view_str
        return view_str

    def __get_view_structure(self, view_dict):
        """
        get the structure of the given view
        :param view_dict: dict, an element of list DeviceState.views
        :return: dict, representing the view structure
        """
        if 'view_structure' in view_dict:
            return view_dict['view_structure']
        width = DeviceState.get_view_width(view_dict)
        height = DeviceState.get_view_height(view_dict)
        class_name = DeviceState.__safe_dict_get(view_dict, 'class', "None")
        children = {}

        root_x = view_dict['bounds'][0][0]
        root_y = view_dict['bounds'][0][1]

        child_view_ids = self.__safe_dict_get(view_dict, 'children')
        if child_view_ids:
            for child_view_id in child_view_ids:
                child_view = self.views[child_view_id]
                child_x = child_view['bounds'][0][0]
                child_y = child_view['bounds'][0][1]
                relative_x, relative_y = child_x - root_x, child_y - root_y
                children["(%d,%d)" % (relative_x, relative_y)] = self.__get_view_structure(child_view)

        view_structure = {
            "%s(%d*%d)" % (class_name, width, height): children
        }
        view_dict['view_structure'] = view_structure
        return view_structure

    @staticmethod
    def __key_if_true(view_dict, key):
        return key if (key in view_dict and view_dict[key]) else ""

    @staticmethod
    def __safe_dict_get(view_dict, key, default=None):
        return view_dict[key] if (key in view_dict) else default

    @staticmethod
    def get_view_center(view_dict):
        """
        return the center point in a view
        @param view_dict: dict, an element of DeviceState.views
        @return: a pair of int
        """
        bounds = view_dict['bounds']
        return (bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2

    @staticmethod
    def get_view_width(view_dict):
        """
        return the width of a view
        @param view_dict: dict, an element of DeviceState.views
        @return: int
        """
        bounds = view_dict['bounds']
        return int(math.fabs(bounds[0][0] - bounds[1][0]))

    @staticmethod
    def get_view_height(view_dict):
        """
        return the height of a view
        @param view_dict: dict, an element of DeviceState.views
        @return: int
        """
        bounds = view_dict['bounds']
        return int(math.fabs(bounds[0][1] - bounds[1][1]))

    def get_all_ancestors(self, view_dict):
        """
        Get temp view ids of the given view's ancestors
        :param view_dict: dict, an element of DeviceState.views
        :return: list of int, each int is an ancestor node id
        """
        result = []
        parent_id = self.__safe_dict_get(view_dict, 'parent', -1)
        if 0 <= parent_id < len(self.views):
            result.append(parent_id)
            result += self.get_all_ancestors(self.views[parent_id])
        return result

    def get_all_children(self, view_dict):
        """
        Get temp view ids of the given view's children
        :param view_dict: dict, an element of DeviceState.views
        :return: set of int, each int is a child node id
        """
        children = self.__safe_dict_get(view_dict, 'children')
        if not children:
            return set()
        children = set(children)
        for child in children:
            children_of_child = self.get_all_children(self.views[child])
            # children.union(children_of_child)
            children = children.union(children_of_child)  # update
        return children

    def get_app_activity_depth(self, app):
        """
        Get the depth of the app's activity in the activity stack
        :param app: App
        :return: the depth of app's activity, -1 for not found
        """
        depth = 0
        for activity_str in self.activity_stack:
            if app.package_name in activity_str:
                return depth
            depth += 1
        return -1

    def get_enabled_view_ids(self):
        """
        Obtain all valid view ids excluding the navigation bar and invalid views with width or height less than or
        equal to 1
        """
        enabled_view_ids = []
        for view_dict in self.views:
            view_size = self.__safe_dict_get(view_dict, 'size').split('*')
            view_w = int(view_size[0])
            view_h = int(view_size[1])
            if self.__safe_dict_get(view_dict, 'enabled') and (view_w > 1 and view_h > 1) and \
                    self.__safe_dict_get(view_dict, 'resource_id') not in \
                    ['android:id/navigationBarBackground', 'android:id/statusBarBackground']:
                enabled_view_ids.append(view_dict['temp_id'])
        return enabled_view_ids

    def get_nav_ids(self):
        """
        Obtain all navigation button ids (if exists)
        """
        nav_ids = set()
        nav_parent_id = None
        enabled_view_ids = self.enabled_view_ids[:]
        enabled_view_ids.reverse()
        device_height = self.device.get_height()
        for view_id in enabled_view_ids:
            view = self.views[view_id]
            view_class = self.__safe_dict_get(view, 'class')
            nav_class = 'LinearLayout' in view_class or 'TabWidget' in view_class or 'ViewGroup' in view_class \
                        or 'RadioGroup' in view_class or 'RecyclerView' in view_class or 'android.view.View' == \
                        view_class or 'CustomItemLayout' in view_class
            child_count = view['child_count']
            if self.__safe_dict_get(view, 'enabled') and nav_class and child_count >= 2:
                children_id = view['children']
                children_class = []
                children_bound_y1 = set()
                children_bound_y2 = set()
                nav_type = set()
                for child_id in children_id:
                    child_view = self.views[child_id]
                    view_size = child_view['size'].split("*")
                    view_w = int(view_size[0])
                    view_h = int(view_size[1])
                    if view_w <= 1 or view_h <= 1:
                        child_count -= 1
                        continue
                    children_class.append(child_view['class'])
                    children_bound_y1.add(child_view['bounds'][0][1])
                    children_bound_y2.add(child_view['bounds'][1][1])
                    nav_type.add(child_view['class'])
                if child_count == 2 and len(nav_type) == 1:
                    if len(children_bound_y1) == 1 and len(children_bound_y2) == 1:  # Horizontal arrangement
                        if 0 <= device_height - children_bound_y2.pop() <= 200 and \
                                0 <= device_height - children_bound_y1.pop() <= 350:  # Near the bottom of the screen
                            nav_parent_id = view_id
                elif child_count > 2 and len(nav_type) <= 2:
                    if len(children_bound_y1) <= 2 and len(children_bound_y2) <= 2:  # Horizontal arrangement
                        min_len = min(len(children_bound_y1), len(children_bound_y2))
                        for index in range(min_len):
                            if 0 <= device_height - children_bound_y2.pop() <= 200 and 0 <= device_height - \
                                    children_bound_y1.pop() <= 350:  # Near the bottom of the screen
                                nav_parent_id = view_id
                            else:
                                nav_parent_id = None
                                break
            if nav_parent_id:
                break
        if nav_parent_id:
            view_parent = self.views[nav_parent_id]
            all_children_ids = self.get_all_children(view_parent)
            nav_ids = all_children_ids
            nav_ids.add(nav_parent_id)
        return nav_ids

    def get_possible_input(self, explored_states=None):
        """
        Get a list of possible input events for this state
        :return: list of InputEvent
        """
        if self.possible_events:
            return [] + self.possible_events

        possible_events = []
        enabled_view_ids = self.enabled_view_ids
        touch_exclude_view_ids = set()

        # Search for confirmation, close, or red packet activation events
        if explored_states is not None:
            specific_events = self.get_specific_input(explored_states)
            if len(specific_events) > 0:
                return specific_events

        # Explore UI states of the app
        for view_id in enabled_view_ids:
            if self.__safe_dict_get(self.views[view_id], 'scrollable'):
                view_class = self.__safe_dict_get(self.views[view_id], 'class')
                if view_class and 'HorizontalScrollView' not in view_class:
                    possible_events.append(ScrollEvent(view=self.views[view_id], direction="DOWN"))
                    # possible_events.append(ScrollEvent(view=self.views[view_id], direction="UP"))
                    # possible_events.append(ScrollEvent(view=self.views[view_id], direction="RIGHT"))
                    # possible_events.append(ScrollEvent(view=self.views[view_id], direction="LEFT"))
                    break

        for view_id in enabled_view_ids:
            if self.__safe_dict_get(self.views[view_id], 'clickable'):
                possible_events.append(TouchEvent(view=self.views[view_id]))
                touch_exclude_view_ids.add(view_id)
                touch_exclude_view_ids = touch_exclude_view_ids.union(self.get_all_children(self.views[view_id]))

        for view_id in enabled_view_ids:
            if self.__safe_dict_get(self.views[view_id], 'long_clickable'):
                possible_events.append(LongTouchEvent(view=self.views[view_id]))
                touch_exclude_view_ids.add(view_id)
                touch_exclude_view_ids = touch_exclude_view_ids.union(self.get_all_children(self.views[view_id]))

        # Search other children nodes that are not clickable
        for view_id in enabled_view_ids:
            if view_id in touch_exclude_view_ids:
                continue
            children = self.__safe_dict_get(self.views[view_id], 'children')
            if children and len(children) > 0:
                continue
            possible_events.append(TouchEvent(view=self.views[view_id]))

        # For old Android navigation bars
        # possible_events.append(KeyEvent(name="MENU"))

        self.possible_events = possible_events
        return [] + possible_events

    def get_search_input(self):
        """
        Get a list of clickable events in this state for searching the navigation bar
        :return: list of InputEvent
        """
        possible_events = []
        enabled_view_ids = self.enabled_view_ids
        touch_exclude_view_ids = set()

        for view_id in enabled_view_ids:
            if self.__safe_dict_get(self.views[view_id], 'clickable'):
                possible_events.append(TouchEvent(view=self.views[view_id]))
                touch_exclude_view_ids.add(view_id)
                touch_exclude_view_ids = touch_exclude_view_ids.union(self.get_all_children(self.views[view_id]))

        # Search other children nodes that are not clickable
        for view_id in enabled_view_ids:
            if view_id in touch_exclude_view_ids:
                continue
            children = self.__safe_dict_get(self.views[view_id], 'children')
            if children and len(children) > 0:
                continue
            possible_events.append(TouchEvent(view=self.views[view_id]))

        self.possible_events = possible_events
        return possible_events

    def get_specific_input(self, explored_states):
        """
        Get a list of specific input events for this state (confirm, close, activate)
        :return: list of InputEvent
        """
        specific_events = []
        enabled_view_ids = self.enabled_view_ids

        # 1 Search for red packet view in the current state and generate the activation event.
        if self.state_str not in explored_states:
            is_red_packet = False
            rp_view_id = None  # ID of red packet view
            reverse_view_ids = self.enabled_view_ids[:]
            reverse_view_ids.reverse()
            # Avoid repeated analysis of different UI views with the same coordinates
            view_coordinates = set()

            for view_id in reverse_view_ids:
                view_class = self.__safe_dict_get(self.views[view_id], 'class')
                view_bounds = self.views[view_id]['bounds']
                # x1-y1-x2-y2
                view_coord = str(view_bounds[0][0]) + '-' + str(view_bounds[0][1]) + '-' + str(
                    view_bounds[1][0]) + '-' + str(view_bounds[1][1])
                # ImageView/Image
                if view_class and "image" in view_class.lower() and view_coord not in view_coordinates:  # Type features
                    view_coordinates.add(view_coord)
                    is_red_packet = self.identify_red_packet(view_id)
                    if is_red_packet:
                        rp_view_id = view_id
                        break
            if not is_red_packet:
                for view_id in reverse_view_ids:
                    view_class = self.__safe_dict_get(self.views[view_id], 'class')
                    view_bounds = self.views[view_id]['bounds']
                    # x1-y1-x2-y2
                    view_coord = str(view_bounds[0][0]) + '-' + str(view_bounds[0][1]) + '-' + str(
                        view_bounds[1][0]) + '-' + str(view_bounds[1][1])
                    # View
                    if view_class and (view_class == "android.view.View" or view_class == "android.view.ViewGroup") and \
                            view_coord not in view_coordinates:  # Type features
                        view_coordinates.add(view_coord)
                        is_red_packet = self.identify_red_packet(view_id)
                        if is_red_packet:
                            rp_view_id = view_id
                            break
            # Expand the search scope
            if not is_red_packet:
                for view_id in reverse_view_ids:
                    view_class = self.__safe_dict_get(self.views[view_id], 'class')
                    view_bounds = self.views[view_id]['bounds']
                    # x1-y1-x2-y2
                    view_coord = str(view_bounds[0][0]) + '-' + str(view_bounds[0][1]) + '-' + str(
                        view_bounds[1][0]) + '-' + str(view_bounds[1][1])
                    # FrameLayout/LinearLayout/RelativeLayout
                    if view_class and "layout" in view_class.lower() and view_coord not in view_coordinates:  # Type feature
                        view_coordinates.add(view_coord)
                        is_red_packet = self.identify_red_packet(view_id)
                        if is_red_packet:
                            rp_view_id = view_id
                            break

            if is_red_packet and rp_view_id is not None:
                self.logger.info("Generating an activation event...")
                event = self.generate_act_event(rp_view_id)
                specific_events.append('activate')
                specific_events.append(event)
                return specific_events

        # 2 Check the confirmation page
        with open('ExploreRecket/resources/keywords/confirm.txt', "r+", encoding='UTF-8') as f:
            confirm_buttons = f.read().split('\n')
        for view_id in enabled_view_ids:
            view = self.views[view_id]
            view_text = view['text']
            if view_text:
                import re
                view_text = re.findall('[\u4e00-\u9fa5a-zA-Z]+', view_text, re.S)
                view_text = ''.join(view_text)
                if len(view['children']) == 0 and (view['clickable'] or self.views[view['parent']]['clickable']):
                    if (view_text.find("同意") != -1 and view_text.find('不同意') == -1 or view_text.find("知道") != -1) \
                            and len(view_text) < 8 or view_text in confirm_buttons:
                        self.logger.info("Find the confirmation button (view = %s)." % view['view_str'])
                        specific_events.append('confirm')
                        specific_events.append(TouchEvent(view=view))
                        return specific_events

        # 3 Check the pop-up or dialog box
        for view_id in enabled_view_ids:
            view = self.views[view_id]
            resource_id = view['resource_id']
            content_desc = view['content_description']
            res_id_status = resource_id is not None and ('close' in resource_id.lower() or 'skip' in resource_id.lower()
                                                         or 'cancel' in resource_id.lower())
            con_desc_status = content_desc is not None and content_desc == '关闭'
            if len(view['children']) == 0 and (res_id_status or con_desc_status):
                self.logger.info("Find the close button (view = %s)." % view['view_str'])
                specific_events.append('close')
                specific_events.append(TouchEvent(view=view))
                return specific_events

        return specific_events

    def get_red_packet_events(self):
        """
        Get a list of related input events that might trigger red packets according to text similarity analysis
        :return: list of InputEvent
        """
        all_possible_events = []
        red_packet_events = []
        other_events = []
        enabled_view_ids = self.enabled_view_ids
        nav_ids = self.get_nav_ids()

        for view_id in enabled_view_ids:
            if self.__safe_dict_get(self.views[view_id], 'clickable') and view_id not in nav_ids:
                file_path = 'ExploreRecket/resources/keywords/red_packet_event.txt'
                with open(file_path, "r+", encoding='UTF-8') as f:
                    event_keywords = f.read().split('\n')
                view_text = self.views[view_id]['text']
                view_desc = self.views[view_id]['content_description']
                if view_text or view_desc:
                    if view_text:
                        text = view_text
                    else:
                        text = view_desc
                # else:
                #     output_dir = os.path.join(self.device.output_dir, "view_temp")
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     status = self.save_view_img(self.views[view_id], output_dir)
                #     if status in [1, -1]:
                #         # Get the text embedded in the view
                #         import time
                #         time.sleep(0.5)
                #         words = self.get_view_text()
                #         words_str = ''
                #         if words is not None:
                #             print('@@@@@@ Words in the view (OCR): ', words)
                #             for word in words:
                #                 words_str += word['words']
                #             text = words_str
                else:
                    all_children_ids = self.get_all_children(self.views[view_id])
                    child_text = ''
                    for child_id in all_children_ids:
                        view_text = self.views[child_id]['text']
                        view_desc = self.views[child_id]['content_description']
                        if view_text:
                            child_text += view_text
                        elif view_desc:
                            child_text += view_desc
                    text = child_text

                strs = re.findall('[\u4e00-\u9fa5]', text)
                text = ''.join(strs)
                if text:
                    # print('view %d text:' % view_id, text)
                    match_status = False
                    for word in event_keywords:
                        if word in text:
                            match_status = True
                            red_packet_events.append(TouchEvent(view=self.views[view_id]))
                            break
                    if not match_status:
                        other_events.append(TouchEvent(view=self.views[view_id]))
                else:
                    other_events.append(TouchEvent(view=self.views[view_id]))

        if len(red_packet_events) > MIN_NUM_EXPLORE_EVENTS:  # Return all possible red packet events
            return red_packet_events
        else:
            all_possible_events = red_packet_events + other_events

        if len(all_possible_events) == 0:
            for view_id in enabled_view_ids:
                if view_id in nav_ids:
                    continue
                children = self.__safe_dict_get(self.views[view_id], 'children')
                if children and len(children) > 0:
                    continue
                all_possible_events.append(TouchEvent(view=self.views[view_id]))

        if len(all_possible_events) > MIN_NUM_EXPLORE_EVENTS:  # Return the first MIN_NUM_EXPLORE_EVENTS events
            all_possible_events = all_possible_events[:MIN_NUM_EXPLORE_EVENTS]

        return all_possible_events

    def identify_red_packet(self, view_id):
        is_red_packet = False
        if self.check_size_position(view_id):  # Size and position features
            output_dir = os.path.join(self.device.output_dir, "candidates/popup")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            status = self.save_view_img(self.views[view_id], output_dir)
            if status == 1:
                self.logger.info("Checking whether the pop-up view is a red packet...")
                if self.check_color():  # Color features
                    is_red_packet = self.check_text()  # Text features
                if not is_red_packet:
                    self.logger.info("The pop-up view is not a red packet.")
                else:
                    self.logger.info("Red packet is found.")
        return is_red_packet

    def check_size_position(self, view_id):
        is_popup = False
        # Size of mobile device screen
        screen_w = self.width
        screen_h = self.height
        # View size
        view_size = self.views[view_id]['size'].split('*')
        view_w = int(view_size[0])
        view_h = int(view_size[1])
        # View bound information
        view_bounds = self.views[view_id]['bounds']
        x1 = view_bounds[0][0]
        x2 = view_bounds[1][0]
        y1 = view_bounds[0][1]
        y2 = view_bounds[1][1]
        # Margin
        margin_top = y1
        margin_bottom = screen_h - y2
        margin_left = x1
        margin_right = screen_w - x2

        # Position and size features (thresholds can be adjusted appropriately)
        if abs(margin_left - margin_right) <= 50 and 0.5 <= view_w / screen_w <= 0.99 and \
                0.25 <= view_h / screen_h <= 0.85 and margin_top > 0 and margin_bottom > 0:
            self.logger.info("This is a pop-up view (class: %s, size: %s)" % (self.__safe_dict_get(self.views[view_id], 'class'),self.views[view_id]['size']))
            is_popup = True
        return is_popup

    def check_color(self):
        is_color = False
        from PIL import Image
        import numpy as np
        image = Image.open(self.view_file_path)
        image_array = np.array(image)
        size = image_array.shape
        height = size[0]
        width = size[1]
        pixel_num = height * width
        bg_pixels = 0
        red_pixels = 0
        yell_pixels = 0
        for h in range(height):
            for w in range(width):
                R = int(image_array[h][w][0])
                G = int(image_array[h][w][1])
                B = int(image_array[h][w][2])
                if max(R, G, B) <= 160 and abs(R - G) <= 10 and abs(R - B) <= 10 and abs(G - B) <= 10:
                    bg_pixels += 1  # Remove the surrounding shadow pixels
                if R >= 180 and G <= 120 and B <= 120:  # Thresholds can be adjusted appropriately
                    red_pixels += 1
                if R >= 200 and G >= 100 and B <= 140:  # Thresholds can be adjusted appropriately
                    yell_pixels += 1
        red_pixels_ratio = round(red_pixels / (pixel_num - bg_pixels + 1), 2)
        yell_pixels_ratio = round(yell_pixels / (pixel_num - bg_pixels + 1), 2)
        if red_pixels_ratio >= 0.3:  # Thresholds can be adjusted (0.2, 0.3, 0.5...)
            # print("Red pixels account for：%f" % red_pixels_ratio)
            is_color = True
        elif yell_pixels_ratio >= 0.3:  # Thresholds can be adjusted (0.2, 0.3, 0.5...)
            # print("Yellow pixels account for：%f" % yell_pixels_ratio)
            is_color = True
        # else:
            # print("Red pixels account for：%f" % red_pixels_ratio)
            # print("Yellow pixels account for：%f" % yell_pixels_ratio)
        return is_color

    def check_text(self):
        is_red_packet = False
        # Get the text embedded in the view
        words = self.get_view_text()
        if words is not None:
            print('OCR results：', words)
            words_str = ''
            for word in words:
                words_str += word['words']

            # Match the most similar red packet text and get the corresponding score
            model = SentenceTransformer('ExploreRecket/resources/paraphrase-multilingual-MiniLM-L12-v2')
            max_score, sim_text = get_sim_score(model, words_str)
            print("Maximum similarity score and most similar red packet text: (%.2f, %s)." % (max_score, sim_text))

            if round(max_score, 2) >= numpy.float32(0.6):
                print("Text matching successful!")
                is_red_packet = True
                # Save the red packet picture locally
                dst_path = os.path.join(self.device.output_dir, "candidates/red_packet/")
                self.copy_file(self.view_file_path, dst_path)
                return is_red_packet

            # Match the open button of the red packet if no text similar to the red packet exists
            if not is_red_packet:
                with open('ExploreRecket/resources/keywords/red_packet_btn.txt', "r", encoding='UTF-8') as f:
                    red_packet_btn_keyword = f.read().split('\n')
                for btn_kwd in red_packet_btn_keyword:
                    for word in words:
                        if word['words'] == btn_kwd:
                            print("The open button matching successful!")
                            is_red_packet = True
                            dst_path = os.path.join(self.device.output_dir, "candidates/red_packet/")
                            self.copy_file(self.view_file_path, dst_path)
                            return is_red_packet
        return is_red_packet

    # Generate the activation event of red packet
    def generate_act_event(self, rp_view_id):
        event = None
        # children = self.__safe_dict_get(self.views[rp_view_id], 'children')
        children = self.get_all_children(self.views[rp_view_id])
        parent = self.__safe_dict_get(self.views[rp_view_id], 'parent')
        brothers = self.__safe_dict_get(self.views[parent], 'children') if (parent and parent != -1) else None

        if self.__safe_dict_get(self.views[rp_view_id], 'clickable'):  # Case1: Red packet view itself
            event = TouchEvent(view=self.views[rp_view_id])
            print("*****Case1:", event)
        elif parent and parent != -1 and self.__safe_dict_get(self.views[parent], 'clickable'):  # Case2: Parent view
            event = TouchEvent(view=self.views[rp_view_id])
            print("*****Case2:", event)
        elif children and len(children) > 0:  # Case3: Child view
            children_sort = sorted(children)
            children_sort.reverse()
            for child in children_sort:
                if self.__safe_dict_get(self.views[child], 'clickable'):
                    view = self.views[child]
                    resource_id = view['resource_id']
                    content_desc = view['content_description']
                    res_id_status = resource_id is not None and (
                                'close' in resource_id.lower() or 'cancel' in resource_id.lower())
                    con_desc_status = content_desc is not None and (content_desc == '关闭' or content_desc == '取消')
                    icon_status = self.reg_close_icon(child)
                    if res_id_status or con_desc_status or icon_status:  # Excluding 'close' button
                        continue
                    else:
                        event = TouchEvent(view=self.views[child])
                        print("*****Case3:", event)
                        break
        elif brothers and len(brothers) > 1:  # Case4: Brother view
            brothers.reverse()
            for brother in brothers:
                if brother != rp_view_id and self.__safe_dict_get(self.views[brother], 'clickable'):
                    view = self.views[brother]
                    resource_id = view['resource_id']
                    content_desc = view['content_description']
                    res_id_status = resource_id is not None and (
                                'close' in resource_id.lower() or 'cancel' in resource_id.lower())
                    con_desc_status = content_desc is not None and (content_desc == '关闭' or content_desc == '取消')
                    icon_status = self.reg_close_icon(brother)
                    if res_id_status or con_desc_status or icon_status:  # Excluding 'close' button
                        continue
                    else:
                        event = TouchEvent(view=self.views[brother])
                        print("*****Case4:", event)
                        break
        if event is None:  # Case5
            event = TouchEvent(view=self.views[rp_view_id])
            print("*****Case5:", event)
        return event

    # Identify the close icon in the red packet by OCR
    def reg_close_icon(self, view_id):
        output_dir = os.path.join(self.device.output_dir, "views/close")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.save_view_img(self.views[view_id], output_dir)
        result = self.get_view_text()
        if result is not None and len(result) == 1:
            text = result[0]['words']
        else:
            return False
        close_icons = ['×', 'x', 'X']
        if text in close_icons:
            return True
        else:
            return False

    # Extract text content in the view by OCR
    def get_view_text(self):
        # Read the image to be identified
        with open(self.view_file_path, 'rb') as fp:
            view_image = fp.read()
        words = None
        # OCR
        result = get_client().basicGeneral(view_image)
        if 'words_result' in result:
            words_num = result['words_result_num']
            if words_num > 0:
                words = result['words_result']
        return words

    # Copy the file to the specified folder
    def copy_file(self, srcfile, dstpath):
        if not os.path.isfile(srcfile):
            self.logger.warning("%s not exist!" % srcfile)
        else:
            fpath, fname = os.path.split(srcfile)
            if not os.path.exists(dstpath):
                os.makedirs(dstpath)
            from datetime import datetime
            tag = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            new_name = 'red-packet-' + tag + '.' + fname.split(".")[-1]
            shutil.copyfile(srcfile, dstpath + new_name)
