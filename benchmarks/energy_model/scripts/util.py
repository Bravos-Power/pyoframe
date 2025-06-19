"""
`mock_snakemake` is inspired from PyPSA-Eur (MIT license, see https://github.com/PyPSA/pypsa-eur/blob/master/scripts/_helpers.py#L476)
"""

from __future__ import annotations

from pathlib import Path


def mock_snakemake(rulename):
    """
    This function is expected to be executed from the 'scripts'-directory of '
    the snakemake project. It returns a snakemake.script.Snakemake object,
    based on the Snakefile.

    If a rule has wildcards, you have to specify them in **wildcards.

    Parameters
    ----------
    rulename: str
        name of the rule for which the snakemake object should be generated
    """
    import os

    import snakemake as sm
    from snakemake.api import Workflow
    from snakemake.common import SNAKEFILE_CHOICES
    from snakemake.script import Snakemake
    from snakemake.settings.types import (
        ConfigSettings,
        DAGSettings,
        ResourceSettings,
        StorageSettings,
        WorkflowSettings,
    )

    script_dir = Path(__file__).parent.resolve()
    root_dir = script_dir.parent.parent.parent
    current_dir = os.getcwd()
    os.chdir(root_dir)

    try:
        for p in SNAKEFILE_CHOICES:
            p = root_dir / p
            if os.path.exists(p):
                snakefile = p
                break
        else:
            raise FileNotFoundError(
                f"Could not find a Snakefile in {root_dir} or its subdirectories."
            )
        workflow = Workflow(
            ConfigSettings(),
            ResourceSettings(),
            WorkflowSettings(),
            StorageSettings(),
            DAGSettings(rerun_triggers=[]),
            storage_provider_settings=dict(),
        )
        workflow.include(snakefile)
        workflow.global_resources = {}
        rule = workflow.get_rule(rulename)
        dag = sm.dag.DAG(workflow, rules=[rule])
        job = sm.jobs.Job(rule, dag)

        def make_accessable(*ios):
            for io in ios:
                for i, _ in enumerate(io):
                    io[i] = os.path.abspath(io[i])

        make_accessable(job.input, job.output, job.log)
        snakemake = Snakemake(
            job.input,
            job.output,
            job.params,
            job.wildcards,
            job.threads,
            job.resources,
            job.log,
            job.dag.workflow.config,
            job.rule.name,
            None,
        )
        # create log and output dir if not existent
        for path in list(snakemake.log) + list(snakemake.output):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    finally:
        os.chdir(current_dir)

    snakemake.mock = True
    return snakemake


if __name__ == "__main__":
    # Example usage
    snakemake = mock_snakemake("process_load_data")
    print(snakemake.input)
    print(snakemake.output)
