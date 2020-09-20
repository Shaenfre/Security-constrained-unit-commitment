import pandas as pd
import numpy as np
import os
import cmath, math


class file_reading:
    def __init__(self, file_name, sheet_name):
        self.excel_sheet_df = pd.read_excel(file_name, sheet_name)

    def open_file(self):
        return self.excel_sheet_df

    # convert a particular column to a numpy array
    def to_numpy(self, col_name):
        return self.excel_sheet_df[col_name].to_numpy()

    # convert all columns into numpy array and store them in a dictionary with column name as the key
    def to_numpy_all(self):
        datasheet_dict = {}
        for col in self.excel_sheet_df.columns:
            datasheet_dict[col] = self.excel_sheet_df[col].to_numpy()
        return datasheet_dict


os.chdir('Project Files')

# Reading bus data
bus_fileReading = file_reading('IEEE9.xlsx', 'Bus Data')
bus_datasheet = bus_fileReading.open_file()

# converting element by element to numpy array
Pg = bus_fileReading.to_numpy('Pg')
Qg = bus_fileReading.to_numpy('Qg')
Pd = bus_fileReading.to_numpy('Pd')
Qd = bus_fileReading.to_numpy('Qd')
type = bus_fileReading.to_numpy('Type')
no_of_buses = len(Pg)
slack_bus_position = np.where(type == 1)
# print(slack_bus_position)

# Reading feeder data
feeder_file_reading = file_reading('IEEE9.xlsx', 'Feeder Data')
# subtracting 1 since in Ymatrix will have 0 based indexing for python
from_bus_col_feeder = feeder_file_reading.to_numpy('From Bus') - 1
to_bus_col_feeder = feeder_file_reading.to_numpy('To Bus') - 1
resistance_bw_frm_and_2_bus_feeder = feeder_file_reading.to_numpy('Resistance')
x_bw_frm_and_2_bus_feeder = feeder_file_reading.to_numpy('Reactance')
chrgng_admtnc_bw_frm_and_2_bus = feeder_file_reading.to_numpy('Charging Admittance')
chrgng_admtnc_bw_frm_and_2_bus = chrgng_admtnc_bw_frm_and_2_bus / 2

# Reading transformer data
transf_file_reading = file_reading('IEEE9.xlsx', 'Transformer Data')
from_bus_col_transf = transf_file_reading.to_numpy('From Bus') - 1
to_bus_col_transf = transf_file_reading.to_numpy('To Bus') - 1
transf_reactance = transf_file_reading.to_numpy('Reactance')
transf_resistance = transf_file_reading.to_numpy('Resistance')
off_nomnl_tap_ratio = transf_file_reading.to_numpy('Off-nominal Tap Ratio')


class convert_to_polar:
    def __init__(self, R_vector, X_vector):
        self.R = R_vector
        self.X = X_vector

    def polar_form(self):
        Y_vector = np.empty(len(self.R), dtype=complex)
        for i in range(len(self.R)):
            Y_vector[i] = 1 / complex(self.R[i], self.X[i])
        return Y_vector


# getting y_vector and charging admmitance vector for feeder
Y_vector_obj = convert_to_polar(resistance_bw_frm_and_2_bus_feeder, x_bw_frm_and_2_bus_feeder)
Y_vector_feeder = Y_vector_obj.polar_form()
# inverting charging admmitance since we are sending impedance to covert_to_polar function
chrgng_admtnc_feedr_obj = convert_to_polar(np.zeros(len(from_bus_col_feeder)),
                                           1 / chrgng_admtnc_bw_frm_and_2_bus)
chrgng_admtnc_feedr = chrgng_admtnc_feedr_obj.polar_form()

# getting y_vector for transformer
Y_vector_obj_transf = convert_to_polar(transf_resistance, transf_reactance)
Y_vector_transf = Y_vector_obj_transf.polar_form()


def form_ybus_with_feeder_data(no_of_bus, from_bus_vector, to_bus_vector, Y_vector, charging_admt_vector):
    Y_bus_matrix = np.zeros((no_of_bus, no_of_bus), dtype=complex)
    for i in range(len(to_bus_vector)):
        Y_bus_matrix[from_bus_vector[i]][from_bus_vector[i]] += Y_vector[i] + charging_admt_vector[i]
        Y_bus_matrix[to_bus_vector[i]][to_bus_vector[i]] += Y_vector[i] + charging_admt_vector[i]
        Y_bus_matrix[from_bus_vector[i]][to_bus_vector[i]] -= Y_vector[i]
        Y_bus_matrix[to_bus_vector[i]][from_bus_vector[i]] -= Y_vector[i]
    return Y_bus_matrix


Y_bus_matrix = form_ybus_with_feeder_data(no_of_buses, from_bus_col_feeder, to_bus_col_feeder
                                          , Y_vector_feeder, chrgng_admtnc_feedr)


# print(Y_bus_matrix)

def modify_ybus_with_transf_reactance(Y_bus_matrix, from_bus_vec, to_bus_vec,
                                      Y_vector, off_nomnl_tap_vec):
    Y_bus_mat = Y_bus_matrix
    for i in range(len(from_bus_vec)):
        Y_bus_mat[from_bus_vec[i]][from_bus_vec[i]] += Y_vector[i]
        Y_bus_mat[from_bus_vec[i]][to_bus_vec[i]] -= Y_vector[i] / off_nomnl_tap_ratio[i]
        Y_bus_mat[to_bus_vec[i]][from_bus_vec[i]] -= Y_vector[i] / off_nomnl_tap_ratio[i]
        Y_bus_mat[to_bus_vec[i]][to_bus_vec[i]] += Y_vector[i] / (off_nomnl_tap_vec[i] ** 2)
    return Y_bus_mat


Y_bus_matrix = modify_ybus_with_transf_reactance(Y_bus_matrix, from_bus_col_transf
                                                   , to_bus_col_transf, Y_vector_transf
                                                   , off_nomnl_tap_ratio)


# print(Y_bus_matrix)

def form_b1_matrix(Y_bus_matrix, slack_bus_position):
    b1 = np.delete(Y_bus_matrix, slack_bus_position, 0)
    b1_modified = np.delete(b1, slack_bus_position, 1)
    b1_modified = b1_modified.imag
    return b1_modified


b1 = form_b1_matrix(Y_bus_matrix, slack_bus_position)
b1_inv = np.linalg.inv(b1)

# print(b1)
# counting the number of PQ buses
unique, count_of_buses = np.unique(type, return_counts=True)
count_dict = dict(zip(unique, count_of_buses))
no_of_PQ_buses = count_dict[3]

def form_b2_matrix(Y_bus_matrix, bus_type_vector, no_of_PQ_buses):
    b2_matrix = np.zeros((no_of_PQ_buses, no_of_PQ_buses), dtype=complex)
    PQ_buses_position = []
    for i in range(len(bus_type_vector)):
        if (bus_type_vector[i] == 3):
            # remember wrt 0 based indexing
            PQ_buses_position.append(i)
    for row in range(len(b2_matrix)):
        for col in range(len(b2_matrix)):
            b2_matrix[row][col] = Y_bus_matrix[PQ_buses_position[row]][PQ_buses_position[col]]
    b2_matrix = b2_matrix.imag
    return b2_matrix


b2_matrix = form_b2_matrix(Y_bus_matrix, type, no_of_PQ_buses)
b2_inv = np.linalg.inv(b2_matrix)
# print(b2_matrix)

# Reading general info datasheet for base mva value
general_info_reading = file_reading('IEEE9.xlsx', 'General Info')
base_mva = general_info_reading.to_numpy('base mva')

P_injected = Pg - Pd
# normalizing P_injected vector
P_injected_normlzd = P_injected / base_mva
Q_injected = Qg - Qd
O_injected_normlzd = Q_injected / base_mva

# Reading PV bus data
PV_bus_reading = file_reading('IEEE9.xlsx', 'PV Bus Data')
specified_voltage_vector = PV_bus_reading.to_numpy('Specified Voltage')
# 0 based indexing
pv_bus_code_vector = PV_bus_reading.to_numpy('Bus Code') - 1
P_min_vector = PV_bus_reading.to_numpy('Pmin')/base_mva
Q_min_vector = PV_bus_reading.to_numpy('Qmin')/base_mva
P_max_vector = PV_bus_reading.to_numpy('Pmax')/base_mva
Q_max_vector = PV_bus_reading.to_numpy('Qmax')/base_mva
pv_bus_sp_volt = PV_bus_reading.to_numpy('Specified Voltage')

#Reading slack bus data
slack_bus_reading = file_reading('IEEE9.xlsx', 'Slack Bus Data')
# 0 based indexing
slack_bus_code = slack_bus_reading.to_numpy('Bus Code') - 1
slack_bus_spcfd_volt = slack_bus_reading.to_numpy('Specified Voltage')

#initializing delta vector
delta_vec = np.zeros((no_of_buses))
#initializing voltage vector
voltage_vec = np.empty((no_of_buses))
for i in range(len(voltage_vec)):
    if i in slack_bus_code:
        voltage_vec[i] = slack_bus_spcfd_volt
    elif i in pv_bus_code_vector:
        index_of_pv_bus = np.where(pv_bus_code_vector == i)
        voltage_vec[i] = pv_bus_sp_volt[index_of_pv_bus]
    else:
        voltage_vec[i] = 1
# print(voltage_vec)

G_matrix = Y_bus_matrix.real
# print(G_matrix)
B_matrix = Y_bus_matrix.imag
# print(B_matrix)

def calc_active_p_vector(voltage_vec, G_matrix, B_matrix, delta_vec, slack_bus_code):
    no_of_bus = len(G_matrix)
    active_power_vec = np.zeros((no_of_bus), dtype=float)
    for i in range(no_of_bus):
        for k in range(no_of_bus):
            if i != slack_bus_code:
                cos_term = G_matrix[i][k] * math.cos(delta_vec[i] - delta_vec[k])
                sin_term = B_matrix[i][k] * math.sin(delta_vec[i] - delta_vec[k])
                summation_term = voltage_vec[k] * (cos_term + sin_term)
                active_power_vec[i] += voltage_vec[i] * summation_term
    active_power_vec = np.delete(active_power_vec, slack_bus_code)
    return active_power_vec

def form_P_specfd_vector(Pg, Pd, type, no_of_bus, base_mva):
    P_specfd_vec = np.zeros(no_of_bus)
    for i in range(no_of_bus):
        P_specfd_vec[i] = Pg[i] - Pd[i]
    # finding index of slack bus
    slack_bus_index = np.where(type == 1)
    P_specfd_vec = np.delete(P_specfd_vec, slack_bus_index)
    P_specfd_vec_mod = P_specfd_vec / base_mva
    return P_specfd_vec_mod


P_specified = form_P_specfd_vector(Pg, Pd, type, no_of_buses, base_mva)


def form_delta_P_by_V_vector(delta_P_vec, voltage_vec, type):
    slack_bus_index = np.where(type == 1)
    new_voltage_vect = np.delete(voltage_vec, slack_bus_index)
    del_P_by_V_vec = delta_P_vec / new_voltage_vect
    return del_P_by_V_vec

PQ_bus_index_list = np.where(type == 3)[0]
# print(PQ_bus_index_list)

delta_V_vector = np.zeros(no_of_PQ_buses)
# print(delta_V_vector)

def form_Q_spcfd_vec(Qg, Qd, no_of_PQ_bus, PQ_bus_index_list, base_mva):
    Q_specfd_vec = np.zeros(no_of_PQ_bus)
    for i in range(no_of_PQ_bus):
        PQ_bus_index = PQ_bus_index_list[i]
        Q_specfd_vec[i] = Qg[PQ_bus_index] - Qd[PQ_bus_index]
    Q_specfd_vec_mod = Q_specfd_vec / base_mva
    return Q_specfd_vec_mod

Q_specfd_vec = form_Q_spcfd_vec(Qg, Qd, no_of_PQ_buses,PQ_bus_index_list, base_mva)
# print(Q_specfd_vec)

def calc_reactive_Q_vector(voltage_vec, G_matrix, B_matrix, delta_vec, PQ_bus_index_list, no_of_PQ_buses, no_of_bus):
    reactive_power_vec = np.zeros(no_of_buses, dtype=object)
    for i in range(no_of_bus):
        for k in range(no_of_bus):
            if i in PQ_bus_index_list:
                cos_term = G_matrix[i][k] * math.sin(delta_vec[i] - delta_vec[k])
                sin_term = B_matrix[i][k] * math.cos(delta_vec[i] - delta_vec[k])
                summation_term = voltage_vec[k] * (cos_term - sin_term)
                reactive_power_vec[i] += voltage_vec[i] * summation_term
    reactive_power_vec = reactive_power_vec[reactive_power_vec != 0]
    return reactive_power_vec
# print(reactive_Q_vec)

def form_delta_Q_by_V_vector(delta_Q_vec, voltage_vec, PQ_bus_index_list):
    voltage_vec = voltage_vec[PQ_bus_index_list]
    del_Q_by_V_vec = delta_Q_vec / voltage_vec
    return del_Q_by_V_vec

no_of_iter = 12
epsilon_p = 10 ** (-5)
epsilon_q = 10 ** (-5)
for i in range(no_of_iter):

    P_calc = calc_active_p_vector(voltage_vec, G_matrix, B_matrix, delta_vec, slack_bus_code)
    delta_P_vec = P_specified - P_calc
    del_P_by_V_vec = form_delta_P_by_V_vector(delta_P_vec, voltage_vec, type)
    del_delta_vector = - np.dot(b1_inv, del_P_by_V_vec)
    max_del_P = max(delta_P_vec)
    print("Max delta p =" + str(max_del_P))

    if max_del_P > epsilon_p:

        # modifying the delta vector
        delta_vec_col_no = 0
        del_delta_vec_col_no = 0

        while (delta_vec_col_no < len(delta_vec)):

            if delta_vec_col_no != slack_bus_code:
                delta_vec[delta_vec_col_no] += del_delta_vector[del_delta_vec_col_no]
                delta_vec_col_no += 1
                del_delta_vec_col_no += 1
            else:
                delta_vec_col_no += 1

        print("Del delta vector iteration no. = " + str(i))
        print(del_delta_vector)
        print("Delta Vector iteration no. = " + str(i))
        print(delta_vec)

    # iteration for modifiying the V vector
    Q_calc = calc_reactive_Q_vector(voltage_vec, G_matrix, B_matrix, delta_vec,
                                    PQ_bus_index_list, no_of_PQ_buses, no_of_buses)
    print("Delta P vec iteration no = " + str(i) + str(Q_calc))
    delta_Q_vec = Q_specfd_vec - Q_calc
    del_Q_by_V_vec = form_delta_Q_by_V_vector(delta_Q_vec, voltage_vec, PQ_bus_index_list)
    del_V_vec = - np.dot(b2_inv, del_Q_by_V_vec)
    max_del_Q = abs(max(np.amax(delta_Q_vec), np.amin(delta_Q_vec), key=abs))

    if max_del_Q > epsilon_q:

        voltage_vec_col_no = 0
        del_V_col_no = 0

        while (voltage_vec_col_no < len(voltage_vec)):
            if voltage_vec_col_no in PQ_bus_index_list:
                voltage_vec[voltage_vec_col_no] += del_V_vec[del_V_col_no]
                voltage_vec_col_no += 1
                del_V_col_no += 1
            else:
                voltage_vec_col_no += 1

        print("Del V vector iteration no. = " + str(i))
        print(del_V_vec)
        print("Voltage vector iteration no. = " + str(i))
        print(voltage_vec)

    # checking if the load flow converged
    if (max_del_Q > epsilon_q or max_del_P > epsilon_p) and i == no_of_iter - 1:
        print("Load flow did not converged")
    elif max_del_P < epsilon_p and max_del_Q < epsilon_q:
        print(f"Load flow converged in {i} of iterations")
        break

