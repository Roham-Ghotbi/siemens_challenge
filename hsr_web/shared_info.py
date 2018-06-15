import Pyro4
"""
control flow (repeats indefinitely)
robot sends img
UI sends back label
robot does some action with label
- can improve with semaphores?

UI control flow:
wait to get img
label img
send back label

robot control flow:
send img
wait to get label
do something with label
"""

@Pyro4.behavior(instance_mode="single")
class server_interface():
    def __init__(self):
        self.curr_img = None
        self.img_ready = False
        self.label_data = None
        self.labeled = False
        self.confidences = None

    @Pyro4.expose
    def is_img_ready(self):
        return self.img_ready

    @Pyro4.expose
    def set_img_ready(self, val):
        self.img_ready = val

    @Pyro4.expose
    def get_img(self):
        return self.curr_img

    @Pyro4.expose
    def set_img(self, img):
        self.curr_img = img

    @Pyro4.expose
    def get_label_data(self):
        return self.label_data

    @Pyro4.expose
    def set_label_data(self, label_data):
        self.label_data = label_data

    @Pyro4.expose
    def is_labeled(self):
        return self.labeled

    @Pyro4.expose
    def set_labeled(self, val):
        self.labeled = val

    @Pyro4.expose
    def display(self):
        return str([self.curr_img, self.img_ready, self.label_data, self.labeled])

    @Pyro4.expose
    def set_confidences(self, confidences):
        self.confidences = confidences

    @Pyro4.expose
    def get_confidences(self):
        return self.confidences

daemon = Pyro4.Daemon()
ns = Pyro4.locateNS()
uri = daemon.register(server_interface)
ns.register("shared.server", uri)

daemon.requestLoop()
