# -*- coding:utf-8 -*-

import argparse


def proc_val(value):
    return value


# aisrv args
def _add_aisrv_args(parser):
    parser.add_argument("--job_master_addr", help="job_master_addr")
    parser.add_argument("--actor_addrs", type=str, help="actor ip_addrs with port")


# actor args
def _add_actor_args(parser):
    parser.add_argument("--job_master_addr", help="job_master_addr")


# learner args
def _add_learner_args(parser):
    parser.add_argument("--job_master_addr", help="job_master_addr")


def cmd_args_parse(svr_name):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )
    parser.add_argument("--conf", default="conf/%s.json" % svr_name, help="config file")

    if svr_name == "aisrv":
        _add_aisrv_args(parser)
    elif svr_name == "actor":
        _add_actor_args(parser)
    elif svr_name == "learner":
        _add_learner_args(parser)
    else:
        RuntimeError("illegal server name %s" % svr_name)

    args, unknowns = parser.parse_known_args()
    for key, value in zip(*[iter(unknowns)] * 2):
        key = key.lstrip("-")
        value = proc_val(value)
        setattr(args, key, value)

    return args
