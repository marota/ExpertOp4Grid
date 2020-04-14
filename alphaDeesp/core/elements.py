"""
This file contains substation elements, ie, Objects: Production, Consumption, Line.
"""


class Production:
    ID = 0

    def __init__(self, busbar_id, value=None):
        # print("Production created...")
        Production.ID += 1
        self.ID = Production.ID
        self.busbar_id = busbar_id
        self.value = value

    def __repr__(self):
        return "<< PRODUCTION Object ID: {}, busbar_id: {}, value: {} >>".format(self.ID, self.busbar_id, self.value)

    @property
    def busbar(self):
        return self.busbar_id

    @busbar.setter
    def busbar(self, new_busbar):
        self.busbar_id = new_busbar


class Consumption:
    ID = 0

    def __init__(self, busbar_id, value=None):
        # print("Consumption created...")
        Consumption.ID += 1
        self.ID = Consumption.ID
        self.busbar_id = busbar_id
        self.value = value

    def __repr__(self):
        return "<< CONSUMPTION Object ID: {}, busbar_id: {}, value: {} >>".format(self.ID, self.busbar_id, self.value)

    @property
    def busbar(self):
        return self.busbar_id

    @busbar.setter
    def busbar(self, new_busbar):
        print("debug inside busbar setter, new busbar =", new_busbar)
        self.busbar_id = new_busbar


class OriginLine:
    ID = 0

    def __init__(self, busbar_id, end_substation_id=None, flow_value=None):
        # print("OriginLine created...")
        OriginLine.ID += 1
        self.ID = OriginLine.ID
        self.busbar_id = busbar_id
        self.end_substation_id = end_substation_id
        self.flow_value = flow_value

    def __repr__(self):
        return "<< ORIGINLINE Object ID: {}, busbar_id: {}," \
               " connected to substation: {}, flow_value: {} >>".format(self.ID, self.busbar_id, self.end_substation_id,
                                                                        self.flow_value)

    @property
    def busbar(self):
        return self.busbar_id

    @busbar.setter
    def busbar(self, new_busbar):
        self.busbar_id = new_busbar


class ExtremityLine:
    ID = 0

    def __init__(self, busbar_id, start_substation_id=None, flow_value=None):
        # print("ExtremityLine created...")
        ExtremityLine.ID += 1
        self.ID = ExtremityLine.ID
        self.busbar_id = busbar_id
        self.start_substation_id = start_substation_id
        self.flow_value = flow_value

    def __repr__(self):
        return "<< EXTREMITYLINE Object ID: {}, busbar_id: {}," \
               " connected to substation: {}, flow_value: {} >>".format(self.ID, self.busbar_id,
                                                                        self.start_substation_id, self.flow_value)

    @property
    def busbar(self):
        return self.busbar_id

    @busbar.setter
    def busbar(self, new_busbar):
        self.busbar_id = new_busbar
