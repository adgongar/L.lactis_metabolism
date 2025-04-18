#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: hamzafaquir and Adrian Gonzalez

Version: 6.0

Program to simulate L. lactis metabolism and to perform parameter estimation.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from time import time
from scipy.integrate import solve_ivp
from scipy.optimize import basinhopping
from multiprocessing import Pool
from datetime import datetime


def import_parameters(path: str) -> tuple:
    """
    Load parameter values and their bounds from a CSV file.

    Reads each line of the file (skipping the header) and builds:
      - a dict mapping parameter names (e.g. 'vmax_gapdh') to float values
      - a dict mapping parameter names to (lower_bound, upper_bound) tuples

    The parameter name is constructed as "<type>_<enzyme>_<metabolite>" or
    "<type>_<enzyme>" if no metabolite is specified.

    Args:
        path (str): Path to the CSV file containing parameters.

    Returns:
        parameters (dict): {param_name: value}
        bounds     (dict): {param_name: (lower_bound, upper_bound)}
    """
    # Create dictionaries
    parameters = {}
    bounds = {}

    # Read each line and extract parameter values and bounds
    # for every parameter. Key structure: param_type_enzyme_metabolite. For
    # parameters that don't depend on metabolites the structure is
    # param_type_enzyme
    with open(path, 'r') as param_file:
        param_file.readline()
        for line in param_file:
            fields = line.strip('\n').split(',')
            enzyme = fields[0].lower()
            param_type = fields[1].lower()
            value = fields[2]
            metabolite = fields[3].lower()
            if metabolite:
                parameters[f'{param_type}_{enzyme}_{metabolite}'] = \
                    float(value)
                bounds[f'{param_type}_{enzyme}_{metabolite}'] = \
                    (float(fields[4]), float(fields[5]))
            else:
                parameters[f'{param_type}_{enzyme}'] = \
                    float(value)
                bounds[f'{param_type}_{enzyme}'] = \
                    (float(fields[4]), float(fields[5]))
    return parameters, bounds


def parameters_to_list(parameters: dict):
    """
    Convert a parameter dictionary into a fixed-order list for the model.

    Extracts each expected parameter by name and returns them in the
    exact sequence required by the kinetic model.

    Args:
        parameters (dict): {param_name: value}

    Returns:
        param_list (list): [vmax_pts, ka_pts_pi, ..., k_o2]
    """

    # PTS
    vmax_pts = parameters['vmax_pts']
    ka_pts_pi = parameters['ka_pts_pi']
    ki_pts_fbp = parameters['ki_pts_fbp']
    km_pts_g6p = parameters['km_pts_g6p']
    km_pts_glucose = parameters['km_pts_glucose']
    km_pts_pep = parameters['km_pts_pep']
    km_pts_pyruvate = parameters['km_pts_pyruvate']

    # ATPase
    vmax_atpase = parameters['vmax_atpase']
    km_atpase_atp = parameters['km_atpase_atp']
    n_atpase = parameters['n_atpase']

    # PI transport
    vmax_pit = parameters['vmax_pit']
    ki_pit_pi = parameters['ki_pit_pi']
    km_pit_adp = parameters['km_pit_adp']
    km_pit_atp = parameters['km_pit_atp']
    km_pit_pi_ext = parameters['km_pit_pi_ext']
    km_pit_pi = parameters['km_pit_pi']

    # PGI #
    vmax_pgi = parameters['vmax_pgi']
    k_eq_pgi = parameters['k_eq_pgi']
    km_pgi_f6p = parameters['km_pgi_f6p']
    km_pgi_g6p = parameters['km_pgi_g6p']

    # PFK #
    vmax_pfk = parameters['vmax_pfk']
    km_pfk_adp = parameters['km_pfk_adp']
    km_pfk_atp = parameters['km_pfk_atp']
    km_pfk_f6p = parameters['km_pfk_f6p']
    km_pfk_fbp = parameters['km_pfk_fbp']
    ki_pfk_atp = parameters['ki_pfk_atp']

    # FBA #
    vmax_fba = parameters['vmax_fba']
    k_eq_fba = parameters['k_eq_fba']
    km_fba_fbp = parameters['km_fba_fbp']
    km_fba_triosep = parameters['km_fba_triosep']

    # GAPDH #
    vmax_gapdh = parameters['vmax_gapdh']
    k_eq_gapdh = parameters['k_eq_gapdh']
    km_gapdh_triosep = parameters['km_gapdh_triosep']
    km_gapdh_pi = parameters['km_gapdh_pi']
    km_gapdh_nad = parameters['km_gapdh_nad']
    km_gapdh_bpg = parameters['km_gapdh_bpg']
    km_gapdh_nadh = parameters['km_gapdh_nadh']

    # ENO #
    k_eq_eno = parameters['k_eq_eno']
    vmax_eno = parameters['vmax_eno']
    km_eno_bpg = parameters['km_eno_bpg']
    km_eno_adp = parameters['km_eno_adp']
    km_eno_pep = parameters['km_eno_pep']
    km_eno_atp = parameters['km_eno_atp']

    # PYK #
    vmax_pyk = parameters['vmax_pyk']
    km_pyk_pep = parameters['km_pyk_pep']
    km_pyk_adp = parameters['km_pyk_adp']
    km_pyk_pyruvate = parameters['km_pyk_pyruvate']
    km_pyk_atp = parameters['km_pyk_atp']
    ki_pyk_pi = parameters['ki_pyk_pi']
    ka_pyk_fbp = parameters['ka_pyk_fbp']
    n_pyk = parameters['n_pyk']

    # LDH #
    vmax_ldh = parameters['vmax_ldh']
    ka_ldh_fbp = parameters['ka_ldh_fbp']
    ki_ldh_pi = parameters['ki_ldh_pi']
    km_ldh_pyruvate = parameters['km_ldh_pyruvate']
    km_ldh_nadh = parameters['km_ldh_nadh']
    km_ldh_lactate = parameters['km_ldh_lactate']
    km_ldh_nad = parameters['km_ldh_nad']

    # ACK #
    vmax_ack = parameters['vmax_ack']
    km_ack_adp = parameters['km_ack_adp']
    km_ack_atp = parameters['km_ack_atp']
    km_ack_acetylcoa = parameters['km_ack_acetylcoa']
    km_ack_acetate = parameters['km_ack_acetate']
    km_ack_coa = parameters['km_ack_coa']
    km_ack_pi = parameters['km_ack_pi']

    # PDC #
    k_eq_pdc = parameters['k_eq_pdc']
    vmax_pdc = parameters['vmax_pdc']
    km_pdc_acetoin = parameters['km_pdc_acetoin']
    km_pdc_pyruvate = parameters['km_pdc_pyruvate']

    # BDH #
    k_eq_bdh = parameters['k_eq_bdh']
    vmax_bdh = parameters['vmax_bdh']
    km_bdh_acetoin = parameters['km_bdh_acetoin']
    km_bdh_butane = parameters['km_bdh_butane']
    km_bdh_nadh = parameters['km_bdh_nadh']
    km_bdh_nad = parameters['km_bdh_nad']

    # trsp #
    vmax_trsp = parameters['vmax_trsp']
    km_trsp_acetoin = parameters['km_trsp_acetoin']
    km_trsp_acetoin_ext = parameters['km_trsp_acetoin_ext']

    # FBPase #
    vmax_fbpase = parameters['vmax_fbpase']
    km_fbpase_f6p = parameters['km_fbpase_f6p']
    km_fbpase_pi = parameters['km_fbpase_pi']
    km_fbpase_fbp = parameters['km_fbpase_fbp']

    # NOX #
    vmax_nox = parameters['vmax_nox']
    km_nox_nadh = parameters['km_nox_nadh']
    km_nox_o2 = parameters['km_nox_o2']
    km_nox_h2o = parameters['km_nox_h2o']
    km_nox_nad = parameters['km_nox_nad']
    k_ind_slope_nox = parameters['k_ind_slope_nox']
    k_ind_res_nox = parameters['k_ind_res_nox']
    k_nox_nis = parameters['k_nox_nis']

    # PDH #
    vmax_pdh = parameters['vmax_pdh']
    km_pdh_pyruvate = parameters['km_pdh_pyruvate']
    km_pdh_nad = parameters['km_pdh_nad']
    km_pdh_nadh = parameters['km_pdh_nadh']
    km_pdh_acetylcoa = parameters['km_pdh_acetylcoa']
    km_pdh_coa = parameters['km_pdh_coa']
    km_pdh_co2 = parameters['km_pdh_co2']

    # O2 #
    k_o2 = parameters['k_o2']

    param_list = [
        # PTS #
        vmax_pts, ka_pts_pi, ki_pts_fbp, km_pts_g6p, km_pts_glucose,
        km_pts_pep, km_pts_pyruvate,

        # ATPase #
        vmax_atpase, km_atpase_atp, n_atpase,

        # PI transport #
        vmax_pit, ki_pit_pi, km_pit_adp, km_pit_atp, km_pit_pi_ext, km_pit_pi,

        # PGI #
        vmax_pgi, k_eq_pgi, km_pgi_f6p, km_pgi_g6p,

        # PFK #
        vmax_pfk, km_pfk_adp, km_pfk_atp, km_pfk_f6p, km_pfk_fbp, ki_pfk_atp,

        # FBA #
        vmax_fba, k_eq_fba, km_fba_fbp, km_fba_triosep,

        # GAPDH #
        vmax_gapdh, k_eq_gapdh, km_gapdh_triosep, km_gapdh_pi, km_gapdh_nad,
        km_gapdh_bpg, km_gapdh_nadh,

        # ENO #
        k_eq_eno, vmax_eno, km_eno_bpg, km_eno_adp, km_eno_pep, km_eno_atp,

        # PYK #
        vmax_pyk, km_pyk_pep, km_pyk_adp, km_pyk_pyruvate, km_pyk_atp,
        ki_pyk_pi, ka_pyk_fbp, n_pyk,

        # LDH #
        vmax_ldh, ka_ldh_fbp, ki_ldh_pi, km_ldh_pyruvate, km_ldh_nadh,
        km_ldh_lactate, km_ldh_nad,

        # ACK #
        vmax_ack, km_ack_adp, km_ack_atp, km_ack_acetylcoa, km_ack_acetate,
        km_ack_coa, km_ack_pi,

        # PDC #
        k_eq_pdc, vmax_pdc, km_pdc_acetoin, km_pdc_pyruvate,

        # BDH #
        k_eq_bdh, vmax_bdh, km_bdh_acetoin, km_bdh_butane, km_bdh_nadh,
        km_bdh_nad,

        # trsp #
        vmax_trsp, km_trsp_acetoin, km_trsp_acetoin_ext,

        # FBPase #
        vmax_fbpase, km_fbpase_f6p, km_fbpase_pi, km_fbpase_fbp,

        # NOX #
        vmax_nox, km_nox_nadh, km_nox_o2, km_nox_h2o, km_nox_nad,
        k_ind_slope_nox, k_ind_res_nox, k_nox_nis,

        # PDH #
        vmax_pdh, km_pdh_pyruvate, km_pdh_nad, km_pdh_nadh, km_pdh_acetylcoa,
        km_pdh_coa, km_pdh_co2,

        # O2 #
        k_o2
        ]

    return param_list


def bounds_to_list(parameters: dict):

    """
    Convert a bounds dictionary into a list of (lower, upper) tuples.

    Args:
        bounds_dict (dict): {param_name: (lower, upper)}

    Returns:
        bounds_list (list): [(lower, upper), ...] in the same order as
        parameters_to_list()
    """

    # PTS #
    vmax_pts = tuple([x for x in parameters['vmax_pts']])
    ka_pts_pi = tuple([x for x in parameters['ka_pts_pi']])
    ki_pts_fbp = tuple([x for x in parameters['ki_pts_fbp']])
    km_pts_g6p = tuple([x for x in parameters['km_pts_g6p']])
    km_pts_glucose = tuple([x for x in parameters['km_pts_glucose']])
    km_pts_pep = tuple([x for x in parameters['km_pts_pep']])
    km_pts_pyruvate = tuple([x for x in parameters['km_pts_pyruvate']])

    # ATPase #
    vmax_atpase = tuple([x for x in parameters['vmax_atpase']])
    km_atpase_atp = tuple([x for x in parameters['km_atpase_atp']])
    n_atpase = tuple([x for x in parameters['n_atpase']])

    # PI transport #
    vmax_pit = tuple([x for x in parameters['vmax_pit']])
    ki_pit_pi = tuple([x for x in parameters['ki_pit_pi']])
    km_pit_adp = tuple([x for x in parameters['km_pit_adp']])
    km_pit_atp = tuple([x for x in parameters['km_pit_atp']])
    km_pit_pi_ext = tuple([x for x in parameters['km_pit_pi_ext']])
    km_pit_pi = tuple([x for x in parameters['km_pit_pi']])

    # PGI #
    vmax_pgi = tuple([x for x in parameters['vmax_pgi']])
    k_eq_pgi = tuple([x for x in parameters['k_eq_pgi']])
    km_pgi_f6p = tuple([x for x in parameters['km_pgi_f6p']])
    km_pgi_g6p = tuple([x for x in parameters['km_pgi_g6p']])

    # PFK #
    vmax_pfk = tuple([x for x in parameters['vmax_pfk']])
    km_pfk_adp = tuple([x for x in parameters['km_pfk_adp']])
    km_pfk_atp = tuple([x for x in parameters['km_pfk_atp']])
    km_pfk_f6p = tuple([x for x in parameters['km_pfk_f6p']])
    km_pfk_fbp = tuple([x for x in parameters['km_pfk_fbp']])
    ki_pfk_atp = tuple([x for x in parameters['ki_pfk_atp']])

    # FBA #
    vmax_fba = tuple([x for x in parameters['vmax_fba']])
    k_eq_fba = tuple([x for x in parameters['k_eq_fba']])
    km_fba_fbp = tuple([x for x in parameters['km_fba_fbp']])
    km_fba_triosep = tuple([x for x in parameters['km_fba_triosep']])

    # GAPDH #
    vmax_gapdh = tuple([x for x in parameters['vmax_gapdh']])
    k_eq_gapdh = tuple([x for x in parameters['k_eq_gapdh']])
    km_gapdh_triosep = tuple([x for x in parameters['km_gapdh_triosep']])
    km_gapdh_pi = tuple([x for x in parameters['km_gapdh_pi']])
    km_gapdh_nad = tuple([x for x in parameters['km_gapdh_nad']])
    km_gapdh_bpg = tuple([x for x in parameters['km_gapdh_bpg']])
    km_gapdh_nadh = tuple([x for x in parameters['km_gapdh_nadh']])

    # ENO #
    k_eq_eno = tuple([x for x in parameters['k_eq_eno']])
    vmax_eno = tuple([x for x in parameters['vmax_eno']])
    km_eno_bpg = tuple([x for x in parameters['km_eno_bpg']])
    km_eno_adp = tuple([x for x in parameters['km_eno_adp']])
    km_eno_pep = tuple([x for x in parameters['km_eno_pep']])
    km_eno_atp = tuple([x for x in parameters['km_eno_atp']])

    # PYK #
    vmax_pyk = tuple([x for x in parameters['vmax_pyk']])
    km_pyk_pep = tuple([x for x in parameters['km_pyk_pep']])
    km_pyk_adp = tuple([x for x in parameters['km_pyk_adp']])
    km_pyk_pyruvate = tuple([x for x in parameters['km_pyk_pyruvate']])
    km_pyk_atp = tuple([x for x in parameters['km_pyk_atp']])
    ki_pyk_pi = tuple([x for x in parameters['ki_pyk_pi']])
    ka_pyk_fbp = tuple([x for x in parameters['ka_pyk_fbp']])
    n_pyk = tuple([x for x in parameters['n_pyk']])

    # LDH #
    vmax_ldh = tuple([x for x in parameters['vmax_ldh']])
    ka_ldh_fbp = tuple([x for x in parameters['ka_ldh_fbp']])
    ki_ldh_pi = tuple([x for x in parameters['ki_ldh_pi']])
    km_ldh_pyruvate = tuple([x for x in parameters['km_ldh_pyruvate']])
    km_ldh_nadh = tuple([x for x in parameters['km_ldh_nadh']])
    km_ldh_lactate = tuple([x for x in parameters['km_ldh_lactate']])
    km_ldh_nad = tuple([x for x in parameters['km_ldh_nad']])

    # ACK #
    vmax_ack = tuple([x for x in parameters['vmax_ack']])
    km_ack_adp = tuple([x for x in parameters['km_ack_adp']])
    km_ack_atp = tuple([x for x in parameters['km_ack_atp']])
    km_ack_acetylcoa = tuple([x for x in parameters['km_ack_acetylcoa']])
    km_ack_acetate = tuple([x for x in parameters['km_ack_acetate']])
    km_ack_coa = tuple([x for x in parameters['km_ack_coa']])
    km_ack_pi = tuple([x for x in parameters['km_ack_pi']])

    # PDC #
    k_eq_pdc = tuple([x for x in parameters['k_eq_pdc']])
    vmax_pdc = tuple([x for x in parameters['vmax_pdc']])
    km_pdc_acetoin = tuple([x for x in parameters['km_pdc_acetoin']])
    km_pdc_pyruvate = tuple([x for x in parameters['km_pdc_pyruvate']])

    # BDH #
    k_eq_bdh = tuple([x for x in parameters['k_eq_bdh']])
    vmax_bdh = tuple([x for x in parameters['vmax_bdh']])
    km_bdh_acetoin = tuple([x for x in parameters['km_bdh_acetoin']])
    km_bdh_butane = tuple([x for x in parameters['km_bdh_butane']])
    km_bdh_nadh = tuple([x for x in parameters['km_bdh_nadh']])
    km_bdh_nad = tuple([x for x in parameters['km_bdh_nad']])

    # trsp #
    vmax_trsp = tuple([x for x in parameters['vmax_trsp']])
    km_trsp_acetoin = tuple([x for x in parameters['km_trsp_acetoin']])
    km_trsp_acetoin_ext = tuple([x for x in parameters['km_trsp_acetoin_ext']])

    # FBPase #
    vmax_fbpase = tuple([x for x in parameters['vmax_fbpase']])
    km_fbpase_f6p = tuple([x for x in parameters['km_fbpase_f6p']])
    km_fbpase_pi = tuple([x for x in parameters['km_fbpase_pi']])
    km_fbpase_fbp = tuple([x for x in parameters['km_fbpase_fbp']])

    # NOX #
    vmax_nox = tuple([x for x in parameters['vmax_nox']])
    km_nox_nadh = tuple([x for x in parameters['km_nox_nadh']])
    km_nox_o2 = tuple([x for x in parameters['km_nox_o2']])
    km_nox_h2o = tuple([x for x in parameters['km_nox_h2o']])
    km_nox_nad = tuple([x for x in parameters['km_nox_nad']])
    k_ind_slope_nox = tuple([x for x in parameters['k_ind_slope_nox']])
    k_ind_res_nox = tuple([x for x in parameters['k_ind_res_nox']])
    k_nox_nis = tuple([x for x in parameters['k_nox_nis']])

    # PDH #
    vmax_pdh = tuple([x for x in parameters['vmax_pdh']])
    km_pdh_pyruvate = tuple([x for x in parameters['km_pdh_pyruvate']])
    km_pdh_nad = tuple([x for x in parameters['km_pdh_nad']])
    km_pdh_nadh = tuple([x for x in parameters['km_pdh_nadh']])
    km_pdh_acetylcoa = tuple([x for x in parameters['km_pdh_acetylcoa']])
    km_pdh_coa = tuple([x for x in parameters['km_pdh_coa']])
    km_pdh_co2 = tuple([x for x in parameters['km_pdh_co2']])

    # O2 #
    k_o2 = tuple([x for x in parameters['k_o2']])

    bounds_list = [
        # PTS #
        vmax_pts, ka_pts_pi, ki_pts_fbp, km_pts_g6p, km_pts_glucose,
        km_pts_pep, km_pts_pyruvate,

        # ATPase #
        vmax_atpase, km_atpase_atp, n_atpase,

        # PI transport #
        vmax_pit, ki_pit_pi, km_pit_adp, km_pit_atp, km_pit_pi_ext, km_pit_pi,

        # PGI #
        vmax_pgi, k_eq_pgi, km_pgi_f6p, km_pgi_g6p,

        # PFK #
        vmax_pfk, km_pfk_adp, km_pfk_atp, km_pfk_f6p, km_pfk_fbp, ki_pfk_atp,

        # FBA #
        vmax_fba, k_eq_fba, km_fba_fbp, km_fba_triosep,

        # GAPDH #
        vmax_gapdh, k_eq_gapdh, km_gapdh_triosep, km_gapdh_pi, km_gapdh_nad,
        km_gapdh_bpg, km_gapdh_nadh,

        # ENO #
        k_eq_eno, vmax_eno, km_eno_bpg, km_eno_adp, km_eno_pep, km_eno_atp,

        # PYK #
        vmax_pyk, km_pyk_pep, km_pyk_adp, km_pyk_pyruvate, km_pyk_atp,
        ki_pyk_pi, ka_pyk_fbp, n_pyk,

        # LDH #
        vmax_ldh, ka_ldh_fbp, ki_ldh_pi, km_ldh_pyruvate, km_ldh_nadh,
        km_ldh_lactate, km_ldh_nad,

        # ACK #
        vmax_ack, km_ack_adp, km_ack_atp, km_ack_acetylcoa, km_ack_acetate,
        km_ack_coa, km_ack_pi,

        # PDC #
        k_eq_pdc, vmax_pdc, km_pdc_acetoin, km_pdc_pyruvate,

        # BDH #
        k_eq_bdh, vmax_bdh, km_bdh_acetoin, km_bdh_butane, km_bdh_nadh,
        km_bdh_nad,

        # trsp #
        vmax_trsp, km_trsp_acetoin, km_trsp_acetoin_ext,

        # FBPase #
        vmax_fbpase, km_fbpase_f6p, km_fbpase_pi, km_fbpase_fbp,

        # NOX #
        vmax_nox, km_nox_nadh, km_nox_o2, km_nox_h2o, km_nox_nad,
        k_ind_slope_nox, k_ind_res_nox, k_nox_nis,

        # PDH #
        vmax_pdh, km_pdh_pyruvate, km_pdh_nad, km_pdh_nadh, km_pdh_acetylcoa,
        km_pdh_coa, km_pdh_co2,

        # O2 #
        k_o2
        ]

    return bounds_list


def L_Lactis_metabolism(time: float, conc0: np.array, parameters: list):
    """
    Compute reaction rates and derivatives for the L. lactis metabolic ODEs.

    This function implements all enzyme rate laws (v_pdh, v_pts, ..., v_nox)
    and returns the time derivatives of each metabolite concentration.

    Args:
        t (float): Current time (ignored in rate equations but required by
                   solve_ivp).
        conc (np.ndarray): Vector of current metabolite concentrations.
        params (list): Model parameters in the order defined by
                       parameters_to_list().

    Returns:
        list: Derivatives [dGlucose, dG6P, ..., dNIS, dCO2].
    """

    # Unpack concentrations
    (Glucose, G6P, F6P, FBP, Triose_P, BPG, PEP, Pyruvate, Lactate, ACETCOA,
        ACETATE, ACETOIN, ACETOIN_ext, BUTANE, P_i, P_i_ex, ATP, ADP, NAD,
        NADH, COA, CO2, NIS) = conc0

    # Save total adenine and nicotinamide pools
    ATP_0 = ATP
    ADP_0 = ADP

    NADH_0 = NADH
    NAD_0 = NAD

    # Unpack parameters
    (
        # PTS #
        vmax_pts, ka_pts_pi, ki_pts_fbp, km_pts_g6p, km_pts_glucose,
        km_pts_pep, km_pts_pyruvate,

        # ATPase #
        vmax_atpase, km_atpase_atp, n_atpase,

        # PI transport #
        vmax_pit, ki_pit_pi, km_pit_adp, km_pit_atp, km_pit_pi_ext, km_pit_pi,

        # PGI #
        vmax_pgi, k_eq_pgi, km_pgi_f6p, km_pgi_g6p,

        # PFK #
        vmax_pfk, km_pfk_adp, km_pfk_atp, km_pfk_f6p, km_pfk_fbp, ki_pfk_atp,

        # FBA #
        vmax_fba, k_eq_fba, km_fba_fbp, km_fba_triosep,

        # GAPDH #
        vmax_gapdh, k_eq_gapdh, km_gapdh_triosep, km_gapdh_pi, km_gapdh_nad,
        km_gapdh_bpg, km_gapdh_nadh,

        # ENO #
        k_eq_eno, vmax_eno, km_eno_bpg, km_eno_adp, km_eno_pep, km_eno_atp,

        # PYK #
        vmax_pyk, km_pyk_pep, km_pyk_adp, km_pyk_pyruvate, km_pyk_atp,
        ki_pyk_pi, ka_pyk_fbp, n_pyk,

        # LDH #
        vmax_ldh, ka_ldh_fbp, ki_ldh_pi, km_ldh_pyruvate, km_ldh_nadh,
        km_ldh_lactate, km_ldh_nad,

        # ACK #
        vmax_ack, km_ack_adp, km_ack_atp, km_ack_acetylcoa, km_ack_acetate,
        km_ack_coa, km_ack_pi,

        # PDC #
        k_eq_pdc, vmax_pdc, km_pdc_acetoin, km_pdc_pyruvate,

        # BDH #
        k_eq_bdh, vmax_bdh, km_bdh_acetoin, km_bdh_butane, km_bdh_nadh,
        km_bdh_nad,

        # trsp #
        vmax_trsp, km_trsp_acetoin, km_trsp_acetoin_ext,

        # FBPase #
        vmax_fbpase, km_fbpase_f6p, km_fbpase_pi, km_fbpase_fbp,

        # NOX #
        vmax_nox, km_nox_nadh, km_nox_o2, km_nox_h2o, km_nox_nad,
        k_ind_slope_nox, k_ind_res_nox, k_nox_nis,

        # PDH #
        vmax_pdh, km_pdh_pyruvate, km_pdh_nad, km_pdh_nadh,
        km_pdh_acetylcoa, km_pdh_coa, km_pdh_co2,

        # O2 #
        k_o2
    ) = parameters

    # Define the rate of change

    # Pyruvate Dehydrogenase (v_pdh)
    v_pdh = (
        (vmax_pdh *
            (Pyruvate / km_pdh_pyruvate) *
            (NAD / km_pdh_nad) *
            (COA / km_pdh_coa)) /
        ((1 + (Pyruvate / km_pdh_pyruvate)) *
            (1 + (NAD / km_pdh_nad)) *
            (1 + (COA / km_pdh_coa)) +
            (1 + (ACETCOA / km_pdh_acetylcoa)) *
            (1 + (CO2 / km_pdh_co2)) *
            (1 + (NADH / km_pdh_nadh)))
    )

    # Phosphotransferase System (v_pts)
    v_pts = (
        (ki_pts_fbp / (ki_pts_fbp + FBP)) *
        (P_i / (ka_pts_pi + P_i)) *
        ((vmax_pts * Glucose * PEP) /
            (km_pts_glucose * km_pts_pep)) /
        ((1 + Glucose / km_pts_glucose) *
            (1 + PEP / km_pts_pep) +
            (1 + G6P / km_pts_g6p) *
            (1 + Pyruvate / km_pts_pyruvate) - 1)
    )

    # Phosphoglucose Isomerase (v_pgi)
    v_pgi = (
        (vmax_pgi *
            (G6P / km_pgi_g6p) - (vmax_pgi / k_eq_pgi) *
            (F6P / km_pgi_f6p)) /
        (1 + G6P / km_pgi_g6p + F6P / km_pgi_f6p)
    )

    # Phosphofructokinase (v_pfk)
    v_pfk = (
        (ki_pfk_atp / (ki_pfk_atp + ATP)) *
        ((vmax_pfk * F6P * ATP) /
            (km_pfk_f6p * km_pfk_atp)) /
        ((1 + F6P / km_pfk_f6p) *
            (1 + ATP / km_pfk_atp) +
            (1 + FBP / km_pfk_fbp) *
            (1 + ADP / km_pfk_adp) - 1)
    )

    # Fructose-1,6-bisphosphate Phosphatase (v_fbpase)
    v_fbpase = (
        (vmax_fbpase *
            FBP / km_fbpase_fbp) /
        ((1 + FBP / km_fbpase_fbp) +
            (1 + F6P / km_fbpase_f6p) *
            (1 + P_i / km_fbpase_pi) - 1)
    )

    # Fructose-1,6-bisphosphate Aldolase (v_fba)
    v_fba = (
        (vmax_fba *
            (FBP / km_fba_fbp) - (vmax_fba / k_eq_fba) *
            ((Triose_P ** 2) / km_fba_fbp)) /
        (1 + (FBP / km_fba_fbp) +
            (Triose_P / km_fba_triosep) +
            ((Triose_P / km_fba_triosep) ** 2))
    )

    # Glyceraldehyde-3-phosphate Dehydrogenase (v_gapdh)
    v_gapdh = (
        ((vmax_gapdh *
            (Triose_P / km_gapdh_triosep) *
            (NAD / km_gapdh_nad) *
            (P_i / km_gapdh_pi)) -
            ((vmax_gapdh / k_eq_gapdh) *
                (BPG / km_gapdh_triosep) *
                (NADH / km_gapdh_nad) *
                (1 / km_gapdh_pi))) /
        ((1 + (Triose_P / km_gapdh_triosep)) *
            (1 + (NAD / km_gapdh_nad)) *
            (1 + (P_i / km_gapdh_pi)) +
            (1 + (BPG / km_gapdh_bpg)) *
            (1 + (NADH / km_gapdh_nadh)) - 1)
    )

    # Enolase (v_eno)
    v_eno = (
        ((vmax_eno *
            (BPG / km_eno_bpg) *
            (ADP / km_eno_adp)) -
            ((vmax_eno / k_eq_eno) *
                (PEP / km_eno_bpg) *
                (ATP / km_eno_adp))) /
        ((1 + (BPG / km_eno_bpg)) *
            (1 + (ADP / km_eno_adp)) +
            (1 + (PEP / km_eno_pep)) *
            (1 + (ATP / km_eno_atp)) - 1)
    )

    # Pyruvate Kinase (v_pyk)
    v_pyk = (
        ((ki_pyk_pi ** n_pyk / (ki_pyk_pi ** n_pyk + P_i ** n_pyk)) *
            (FBP / (ka_pyk_fbp + FBP)) *
            (vmax_pyk *
                (PEP / km_pyk_pep) *
                (ADP / km_pyk_adp))) /
        ((1 + PEP / km_pyk_pep) *
            (1 + ADP / km_pyk_adp) +
            (1 + Pyruvate / km_pyk_pyruvate) *
            (1 + ATP / km_pyk_atp) - 1)
    )

    # L-lactate Dehydrogenase (v_ldh)
    v_ldh = (
        ((FBP / (ka_ldh_fbp + FBP)) *
            (ki_ldh_pi / (ki_ldh_pi + P_i)) *
            (vmax_ldh *
                (Pyruvate / km_ldh_pyruvate) *
                (NADH / km_ldh_nadh))) /
        ((1 + Pyruvate / km_ldh_pyruvate) *
            (1 + NADH / km_ldh_nadh) +
            (1 + Lactate / km_ldh_lactate) *
            (1 + NAD / km_ldh_nad) - 1)
    )

    # Acetate Kinase (v_ack)
    v_ack = (
        (vmax_ack *
            (ACETCOA / km_ack_acetylcoa) *
            (ADP / km_ack_adp) *
            (P_i / km_ack_pi)) /
        ((1 + P_i / km_ack_pi) *
            (1 + ACETCOA / km_ack_acetylcoa) *
            (1 + ADP / km_ack_adp) +
            (1 + ACETATE / km_ack_acetate) *
            (1 + ATP / km_ack_atp) *
            (1 + COA / km_ack_coa) - 1)
    )

    # Pyruvate Decarboxylase (v_pdc)
    v_pdc = (
        (vmax_pdc *
            (Pyruvate / km_pdc_pyruvate) ** 2 -
            (vmax_pdc / k_eq_pdc) *
            (ACETOIN / km_pdc_pyruvate)) /
        ((1 + Pyruvate / km_pdc_pyruvate +
            (Pyruvate / km_pdc_pyruvate)**2) +
            (1 + ACETOIN / km_pdc_acetoin) - 1)
    )

    # 2,3-Butanediol Dehydrogenase (v_bdh)
    v_bdh = (
        ((vmax_bdh *
            (ACETOIN / km_bdh_acetoin) *
            (NADH / km_bdh_nadh)) -
            ((vmax_bdh / k_eq_bdh) *
                (BUTANE / km_bdh_acetoin) *
                (NAD / km_bdh_nadh))) /
        ((1 + ACETOIN / km_bdh_acetoin) *
            (1 + NADH / km_bdh_nadh) +
            (1 + BUTANE / km_bdh_butane) *
            (1 + NAD / km_bdh_nad) - 1)
    )

    # Acetoin Transport (v_trsp)
    v_trsp = (
        (vmax_trsp *
            (ACETOIN / km_trsp_acetoin)) /
        ((1 + ACETOIN / km_trsp_acetoin) +
            (1 + ACETOIN_ext / km_trsp_acetoin_ext) - 1)
    )

    # Phosphate Transport (v_pit)
    v_pit = (
        ((ki_pit_pi / (ki_pit_pi + P_i)) *
            (vmax_pit *
                (P_i_ex / km_pit_pi_ext) *
                (ATP / km_pit_atp))) /
        ((1 + P_i_ex / km_pit_pi_ext) *
            (1 + ATP / km_pit_atp) +
            (1 + P_i / km_pit_pi +
                (P_i / km_pit_pi) ** 2) *
            (1 + ADP / km_pit_adp) - 1)
    )

    # ATPase (v_atpase)
    v_atpase = (
        (vmax_atpase *
            (ATP / km_atpase_atp) ** n_atpase) /
        (1 + (ATP / km_atpase_atp) ** n_atpase)
    )

    # NADH Oxidase Regulation (Nox_regu)
    Nox_regu = (
        k_ind_slope_nox *
        k_nox_nis *
        NIS + k_ind_res_nox *
        k_nox_nis
    )

    # NOX (v_nox)
    v_nox = (
        Nox_regu *
        ((NADH / km_nox_nadh) ** 2 *
            (k_o2 / km_nox_o2)) /
        ((1 + (NADH / km_nox_nadh) + (NADH / km_nox_nadh)**2) *
            (1 + (k_o2 / km_nox_o2)) +
            (1 + (NAD / km_nox_nad) +
             (NAD / km_nox_nad)**2) - 1)
    )

    # Ensure ATP/ADP and NAD/NADH mass conservation
    ADP = ADP_0 + ATP_0 - ATP
    NADH = NADH_0 + NAD_0-NAD

    # ODEs #

    dGlucose = -v_pts

    dG6P = v_pts - v_pgi

    dF6P = v_pgi - v_pfk + v_fbpase

    dFBP = v_pfk - v_fbpase - v_fba

    dTriose_P = 2 * v_fba - v_gapdh

    dBPG = v_gapdh - v_eno

    dPEP = -v_pts + v_eno - v_pyk

    dPyruvate = v_pts - v_pdh - 2 * v_pdc + v_pyk - v_ldh

    dACETCOA = v_pdh - v_ack

    dACETOIN = v_pdc - v_bdh - v_trsp

    dACETOIN_ext = v_trsp

    dBUTANE = v_bdh

    dLactate = v_ldh

    dACETATE = v_ack

    dP_i = v_fbpase - v_gapdh + 2 * v_pit + v_atpase - v_ack

    dP_i_ex = - v_pit

    dATP = - v_atpase + v_ack - v_pit - v_pfk + v_eno + v_pyk

    dNAD = v_bdh - v_gapdh + v_ldh + 2 * v_nox - v_pdh

    dCOA = v_ack - v_pdh

    dADP = -dATP

    dNADH = - dNAD

    dNIS = 0

    dCO2 = v_pdh + 2 * v_pdc

    return [dGlucose, dG6P, dF6P, dFBP, dTriose_P, dBPG,
            dPEP, dPyruvate, dLactate, dACETCOA, dACETATE,
            dACETOIN, dACETOIN_ext, dBUTANE, dP_i, dP_i_ex,
            dATP, dADP, dNAD, dNADH, dCOA, dCO2, dNIS]


def get_param_index(param_list: list, full_parameters: list) -> list:
    """
    Find the indices of given parameter names within the full parameter list.

    Args:
        param_list (list): List of parameter names to locate.
        full_list (list): Complete ordered list of all parameter names.

    Returns:
        index_list (list): Indices corresponding to each name in param_list.
    """
    index_list = []
    for param in param_list:
        i = full_parameters.index(param)
        index_list.append(i)
    return index_list


def Plot_results(Y: list, t: np.array):
    """
    Plot time courses of all metabolite concentrations.

    Displays a 6x4 grid of subplots, one per metabolite, versus time in
    minutes.

    Args:
        Y (np.ndarray): Shape (len(t), n_metabolites) matrix of concentrations.
        t (np.ndarray): Time vector (seconds).
    """

    # Convert time to minutes
    t = t / 60

    fig, axs = plt.subplots(6, 4, figsize=(25, 15))

    fig.suptitle('Evolution of concentrations')

    axs[0, 0].plot(t, Y[:, 0])
    axs[0, 0].set(xlabel='Time(min)', ylabel='Glucose')

    axs[0, 1].plot(t, Y[:, 1])
    axs[0, 1].set(xlabel='Time(min)', ylabel='G6P')

    axs[0, 2].plot(t, Y[:, 2])
    axs[0, 2].set(xlabel='Time(min)', ylabel='F6P')

    axs[0, 3].plot(t, Y[:, 3])
    axs[0, 3].set(xlabel='Time(min)', ylabel='FBP')

    axs[1, 0].plot(t, Y[:, 4])
    axs[1, 0].set(xlabel='Time(min)', ylabel='G3P')

    axs[1, 1].plot(t, Y[:, 5])
    axs[1, 1].set(xlabel='Time(min)', ylabel='BPG')

    axs[1, 2].plot(t, Y[:, 6])
    axs[1, 2].set(xlabel='Time(min)', ylabel='PEP')

    axs[1, 3].plot(t, Y[:, 7])
    axs[1, 3].set(xlabel='Time(min)', ylabel='Pyruvate')

    axs[2, 0].plot(t, Y[:, 8])
    axs[2, 0].set(xlabel='Time()', ylabel='Lactate')

    axs[2, 1].plot(t, Y[:, 9])
    axs[2, 1].set(xlabel='Time()', ylabel='AcetCoa')

    axs[2, 3].plot(t, Y[:, 10])
    axs[2, 3].set(xlabel='Time()', ylabel='ACETATE')

    axs[3, 0].plot(t, Y[:, 11])
    axs[3, 0].set(xlabel='Time()', ylabel='Acetoin')

    axs[3, 1].plot(t, Y[:, 12])
    axs[3, 1].set(xlabel='Time()', ylabel='Acetoin_ext')

    axs[3, 2].plot(t, Y[:, 13])
    axs[3, 2].set(xlabel='Time()', ylabel='BUTANE')

    axs[3, 3].plot(t, Y[:, 14])
    axs[3, 3].set(xlabel='Time()', ylabel='Phosphate')

    axs[4, 0].plot(t, Y[:, 15])
    axs[4, 0].set(xlabel='Time()', ylabel='Phosphate_ext')

    axs[4, 1].plot(t, Y[:, 16], label='ATP')
    axs[4, 1].plot(t, Y[:, 17], label='ADP')
    axs[4, 1].set(xlabel='Time()', ylabel='ATP/ADP')
    axs[4, 1].legend()

    axs[4, 2].plot(t, Y[:, 18], label='NAD')
    axs[4, 2].plot(t, Y[:, 19], label='NADH')
    axs[4, 2].set(xlabel='Time()', ylabel='NADH/NAD')
    axs[4, 2].legend()

    axs[4, 3].plot(t, Y[:, 20])
    axs[4, 3].set(xlabel='Time()', ylabel='COA')
    axs[4, 3].legend()

    axs[5, 0].plot(t, Y[:, 21])
    axs[5, 0].set(xlabel='Time()', ylabel='CO2')
    axs[5, 0].legend()

    axs[5, 1].plot(t, Y[:, 22], label='NIS')
    axs[5, 1].set(xlabel='Time()', ylabel='NIS')
    axs[5, 1].legend()

    plt.show()


def Plot_sim(lactate, acetate, sim_lactate, sim_acetate, exp_data):

    """
    Compare experimental and simulated lactate/acetate at steady state.

    Plots means ± standard error for experiments and overlays simulation points
    as a function of nisin concentration.

    Args:
        lactate   (pd.DataFrame): Experimental lactate with
                                  columns ['mean','var'].
        acetate   (pd.DataFrame): Experimental acetate with
                                  columns ['mean','var'].
        sim_lac   (np.ndarray): Simulated lactate at each nisin condition.
        sim_ace   (np.ndarray): Simulated acetate at each nisin condition.
        exp_data  (pd.DataFrame): Raw experimental data (for scatter plots).
    """

    lactate_mean = np.array(lactate['mean'])
    lactate_var = np.array(lactate['var'])
    lactate_err = np.sqrt(lactate_var)

    acetate_mean = np.array(acetate['mean'])
    acetate_var = np.array(acetate['var'])
    acetate_err = np.sqrt(acetate_var)

    nis = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
    fig, axs = plt.subplots(2, 2, figsize=(25, 15))

    axs[0, 0].errorbar(nis, lactate_mean, yerr=lactate_err, fmt='o',
                       color='blue', ecolor='blue', elinewidth=2, capsize=4,
                       label='Experimental lactate')
    axs[0, 0].scatter(nis, sim_lactate, color='red',
                      label='Simultated lactate')
    axs[0, 0].set_xlabel('Nisin')
    axs[0, 0].set_ylabel('Lactate (mM)')

    axs[0, 1].errorbar(nis, acetate_mean, yerr=acetate_err, fmt='o',
                       color='blue', ecolor='blue', elinewidth=2, capsize=4,
                       label='Experimental acetate')
    axs[0, 1].scatter(nis, sim_acetate, color='red',
                      label='Simultated acetate')
    axs[0, 1].set_xlabel('Nisin')
    axs[0, 1].set_ylabel('Acetate (mM)')

    axs[1, 0].scatter(exp_data['Nisina (U/mL)'], exp_data['Lactato (mM)'])
    axs[1, 0].set(xlabel='Nisina (U/mL)', ylabel='Lactato (mM)')

    axs[1, 1].scatter(exp_data['Nisina (U/mL)'], exp_data['Acetato (mM)'])
    axs[1, 1].set(xlabel='Nisina (U/mL)', ylabel='Acetato (mM)')

    plt.show()


def nisin_sim(IC: list, time: np.array, parameters: list, plot=True):
    """
    Simulate metabolism across a range of nisin concentrations.

    For each nisin level, integrates the ODEs to steady state and collects
    final lactate and acetate concentrations.

    Args:
        IC (list): Initial metabolite concentrations.
        t (np.ndarray): Time vector for integration.
        params (list): Model parameters.
        plot (bool): If True, call Plot_results() for each run.

    Returns:
        tuple:
            - lactate_final (np.ndarray): Final lactate for each nisin.
            - acetate final (np.ndarray): Final acetate for each nisin.
    """

    # Define nisin conditions
    nis = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])

    lactate_final = np.zeros(len(nis))
    acetate_final = np.zeros(len(nis))

    # Set solver parameters
    solver_kwargs = dict(method='BDF', t_eval=time, args=(parameters,))

    # Simulate L. lactis metabolism for every nisin condition
    for i, val in enumerate(nis):
        IC_local = IC.copy()
        IC_local[22] = val
        sol = solve_ivp(L_Lactis_metabolism, [time[0], time[-1]], IC_local,
                        **solver_kwargs)
        X = sol.y.T
        if plot:
            Plot_results(X, time)
        lactate_final[i] = X[-1, 8]
        acetate_final[i] = X[-1, 10]
    return lactate_final, acetate_final


def simulate_condition(args: list):
    """
    Helper for parallel simulation of a single nisin condition.

    Args:
        args (tuple): (nisin_value, IC, t, params, solver_kwargs)

    Returns:
        (lactate_ss, acetate_ss): Steady-state concentrations.
    """
    nis_val, IC, time, parameters_local, solver_kwargs = args
    IC_local = IC.copy()
    IC_local[22] = nis_val
    sol = solve_ivp(L_Lactis_metabolism, [time[0], time[-1]], IC_local,
                    **solver_kwargs)
    X = sol.y.T
    # Return steady-state concentrations of lactate and acetate
    lactate_final = X[-1, 8]
    acetate_final = X[-1, 10]

    return lactate_final, acetate_final


def Objetive_function(parameters_estimate: list, index_params: list, IC: list,
                      time: np.array, parameters_original: list,
                      lactate_mean: np.array, lactate_var: np.array,
                      acetate_mean: np.array, acetate_var: np.array):
    """
    Compute the objective function for parameter estimation.

    Updates the selected parameters with values in `x`, runs parallel
    steady-state simulations, and returns the sum of squared errors
    normalized by experimental variances.

    Args:
        parameters_estimate (list): Initial guesses for parameters to optimize.
        index_params (list): Indices of parameters in the full list.
        IC (list): Initial conditions.
        time (np.ndarray): Time vector.
        params_original (list): Original full parameter list.
        lactate_mean (np.ndarray): Experimental lactate means.
        lactate_var  (np.ndarray): Experimental lactate variances.
        acetate_mean (np.ndarray): Experimental acetate means.
        acetate_var  (np.ndarray): Experimental acetate variances.

    Returns:
        float: Total weighted squared error.
    """
    # Make a copy of parameters
    parameters_local = parameters_original.copy()
    for i, idx in enumerate(index_params):
        parameters_local[idx] = parameters_estimate[i]

    # Set solver parameters
    nis = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
    solver_kwargs = dict(method='BDF', t_eval=time, args=(parameters_local,))

    # Parameters for every nisin condition
    args_list = [(val, IC, time, parameters_local, solver_kwargs)
                 for val in nis]

    # Parallel execution
    with Pool() as pool:
        results = pool.map(simulate_condition, args_list)

    # Get results
    lactate_final = np.array([res[0] for res in results])
    acetate_final = np.array([res[1] for res in results])

    # Calculate standard deviation
    lactate_std = np.sqrt(lactate_var)
    acetate_std = np.sqrt(acetate_var)

    # Calculate error
    err_lactate = ((lactate_final - lactate_mean) / lactate_std) ** 2
    err_acetate = ((acetate_final - acetate_mean) / acetate_std) ** 2
    total_error = np.sum(err_lactate + err_acetate)

    return total_error


def basinhopping_callback(x, f, accept):
    """
    Callback to record basinhopping progress at each iteration.

    Writes current parameters `x` and objective value `f` to an output file,
    appends to the history list, and prints iteration count.

    Args:
        x (array): Current parameter vector.
        f (float): Objective function value.
        accepted (bool): Whether the step was accepted.
    """

    global f_out
    global progress
    global optimization_history

    # Save progress in a file
    optimization_history.append(f)
    text = '\t'.join(map(str, x))
    out_text = f'{text}\t{f}\n'
    f_out.write(out_text)
    f_out.flush()
    print(f'Terminada iteración {progress}')
    progress += 1


def main():
    """
    Main entry point for parameter optimization workflow.

    1. Load experimental data and compute means/variances.
    2. Import model parameters and convert to list form.
    3. Run initial simulation and plot results.
    4. If enabled, optimize selected parameters via basinhopping,
       log progress, and plot convergence.
    5. Simulate with optimized parameters and compare again.
    """
    global optimization_history
    global progress
    global f_out

    progress = 1
    optimization_history = []

    # Read experimental data
    exp_data = pd.read_csv('datos_finales_4rep.txt', sep=' ')
    lactate = exp_data.groupby('Nisina (U/mL)')['Lactato (mM)'].agg(['mean', 'var']).reset_index()
    lactate_mean = np.array(lactate['mean'])
    lactate_var = np.array(lactate['var'])

    acetate = exp_data.groupby('Nisina (U/mL)')['Acetato (mM)'].agg(['mean', 'var']).reset_index()
    acetate_mean = np.array(acetate['mean'])
    acetate_var = np.array(acetate['var'])

    # Load parameters
    param_dict, bounds_dict = import_parameters('parametros_input.csv')
    parameters = parameters_to_list(param_dict)
    bounds = bounds_to_list(bounds_dict)

    parameter_list = [
        'vmax_pts', 'ka_pts_pi', 'ki_pts_fbp', 'km_pts_g6p', 'km_pts_glucose',
        'km_pts_pep', 'km_pts_pyruvate',
        'vmax_atpase', 'km_atpase_atp', 'n_atpase',
        'vmax_pit', 'ki_pit_pi', 'km_pit_adp', 'km_pit_atp', 'km_pit_pi_ext',
        'km_pit_pi',
        'vmax_pgi', 'k_eq_pgi', 'km_pgi_f6p', 'km_pgi_g6p',
        'vmax_pfk', 'km_pfk_adp', 'km_pfk_atp', 'km_pfk_f6p', 'km_pfk_fbp',
        'ki_pfk_atp',
        'vmax_fba', 'k_eq_fba', 'km_fba_fbp', 'km_fba_triosep',
        'vmax_gapdh', 'k_eq_gapdh', 'km_gapdh_triosep', 'km_gapdh_pi',
        'km_gapdh_nad', 'km_gapdh_bpg', 'km_gapdh_nadh',
        'k_eq_eno', 'vmax_eno', 'km_eno_bpg', 'km_eno_adp', 'km_eno_pep',
        'km_eno_atp',
        'vmax_pyk', 'km_pyk_pep', 'km_pyk_adp', 'km_pyk_pyruvate',
        'km_pyk_atp', 'ki_pyk_pi', 'ka_pyk_fbp', 'n_pyk',
        'vmax_ldh', 'ka_ldh_fbp', 'ki_ldh_pi', 'km_ldh_pyruvate',
        'km_ldh_nadh', 'km_ldh_lactate', 'km_ldh_nad',
        'vmax_ack', 'km_ack_adp', 'km_ack_atp', 'km_ack_acetylcoa',
        'km_ack_acetate', 'km_ack_coa', 'km_ack_pi',
        'k_eq_pdc', 'vmax_pdc', 'km_pdc_acetoin', 'km_pdc_pyruvate',
        'k_eq_bdh', 'vmax_bdh', 'km_bdh_acetoin', 'km_bdh_butane',
        'km_bdh_nadh', 'km_bdh_nad',
        'vmax_trsp', 'km_trsp_acetoin', 'km_trsp_acetoin_ext',
        'vmax_fbpase', 'km_fbpase_f6p', 'km_fbpase_pi', 'km_fbpase_fbp',
        'vmax_nox', 'km_nox_nadh', 'km_nox_o2', 'km_nox_h2o', 'km_nox_nad',
        'k_ind_slope_nox', 'k_ind_res_nox', 'k_nox_nis',
        'vmax_pdh', 'km_pdh_pyruvate', 'km_pdh_nad', 'km_pdh_nadh',
        'km_pdh_acetylcoa', 'km_pdh_coa', 'km_pdh_co2',
        'k_o2'
    ]

    # Set initial conditions
    Glucose_0 = 27.75  # 0.5 %
    G6P_0 = 0
    F6P_0 = 0
    FBP_0 = 15.3
    Triose_P_0 = 0
    BPG_0 = 1.26
    PEP_0 = 2.48
    Pyruvate_0 = 0
    ACETCOA_0 = 0
    ACETOIN_0 = 0
    ACETOIN_ext_0 = 0
    BUTANE_0 = 0
    LACTATE_0 = 0
    ACETATE_0 = 0
    P_i_0 = 38.26
    P_i_ex_0 = 40
    ATP_0 = 4.89
    NAD_0 = 4.67
    COA_0 = 1
    NADH_0 = 2.03e-6
    ADP_0 = 20.39
    NIS_0 = 0.25
    CO2_0 = 0

    IC = [Glucose_0, G6P_0, F6P_0, FBP_0, Triose_P_0, BPG_0, PEP_0,
          Pyruvate_0, LACTATE_0, ACETCOA_0, ACETATE_0, ACETOIN_0,
          ACETOIN_ext_0, BUTANE_0, P_i_0, P_i_ex_0, ATP_0, ADP_0, NAD_0,
          NADH_0, COA_0, CO2_0, NIS_0]

    tf = 18000
    t = np.linspace(0, tf, 1000)

    # Set parameters to optimize
    params_to_optimize = [
        'vmax_pdh',
        'vmax_pdc',
        'km_pdh_nad',
        'km_pdh_nadh',
        'km_ack_pi',
        'km_nox_nad',
        'k_nox_nis',
        'k_o2'
    ]

    index_params = get_param_index(params_to_optimize, parameter_list)
    initial_guess = [parameters[i] for i in index_params]
    bounds_guess = [bounds[i] for i in index_params]

    # Initial simulation
    sim_lactate, sim_acetate = nisin_sim(IC, t, parameters, plot=True)
    Plot_sim(lactate, acetate, sim_lactate, sim_acetate, exp_data)

    # Optimization
    optimizar = True
    if optimizar:
        date = datetime.now()
        form_date = date.strftime('%d-%m-%Y_%H%M%S')
        f_out = open(f'iteraciones_optimizacion_{form_date}.csv', 'w')
        f_out.write('\t'.join(params_to_optimize) + '\tObj.Func\n')

        t0 = time()
        minimizer_kwargs = {
            "args": (index_params, IC, t, parameters, lactate_mean,
                     lactate_var, acetate_mean, acetate_var),
            "bounds": bounds_guess
        }

        optim_result = basinhopping(
            Objetive_function, initial_guess,
            minimizer_kwargs=minimizer_kwargs, niter=1,
            callback=basinhopping_callback, T=1
        )

        t1 = time()
        print(f'The parameters values that optimize the model are: \
            {optim_result.x}. El proceso ha tardado {t1 - t0}s')

        f_out.close()

        # Plot convergence
        plt.figure(figsize=(10, 5))
        plt.plot(optimization_history, label='Objective Function Value')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Function Value')
        plt.title('Convergence of Optimization')
        plt.legend()
        plt.show()

        # Simulation with optimized parameters
        parameters_final = parameters.copy()
        for i, idx in enumerate(index_params):
            parameters_final[idx] = optim_result.x[i]
        sim_lactate_opt, sim_acetate_opt = \
            nisin_sim(IC, t, parameters_final, plot=True)
        Plot_sim(lactate, acetate, sim_lactate_opt, sim_acetate_opt, exp_data)


if __name__ == '__main__':
    main()
