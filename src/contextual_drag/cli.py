from __future__ import annotations

import sys

import scriptconfig as scfg

from contextual_drag.data.aggregate_crux_data_cli import AggregateCruxDataCLI
from contextual_drag.data.aggregate_data_cli import AggregateDataCLI
from contextual_drag.data.aggregate_data_iterative_cli import AggregateDataIterativeCLI
from contextual_drag.data.initial_sampling_postprocess_cli import InitialSamplingPostprocessCLI
from contextual_drag.data.minimal_aggregate_flatten_cli import MinimalAggregateFlattenCLI
from contextual_drag.data.stage1_postprocess_iterative_cli import Stage1PostprocessIterativeCLI
from contextual_drag.evaluation.crux.eval_cli import EvalCruxCLI
from contextual_drag.evaluation.math.eval_cli import EvalMathCLI
from contextual_drag.inference.vllm_cli import InferenceListModelsCLI, InferenceRunCLI


class InferenceCLI(scfg.ModalCLI):
    run = InferenceRunCLI
    list_models = InferenceListModelsCLI


class EvalCLI(scfg.ModalCLI):
    math = EvalMathCLI
    crux = EvalCruxCLI


class DataCLI(scfg.ModalCLI):
    initial_sampling_postprocess = InitialSamplingPostprocessCLI
    minimal_aggregate_flatten = MinimalAggregateFlattenCLI
    aggregate = AggregateDataCLI
    aggregate_crux = AggregateCruxDataCLI
    aggregate_iterative = AggregateDataIterativeCLI
    stage1_postprocess_iterative = Stage1PostprocessIterativeCLI


class ContextualDragCLI(scfg.ModalCLI):
    inference = InferenceCLI
    eval = EvalCLI
    data = DataCLI


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    modal_argv = argv or ["--help"]
    result = ContextualDragCLI.main(argv=modal_argv, _noexit=True)
    if result == 1 and (not argv or "--help" in argv or "-h" in argv):
        return 0
    return int(result or 0)
