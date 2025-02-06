import json
import time
import openai
import argparse
import os

from loguru import logger
from dataclasses import dataclass, field
import traceback

from lean_dojo import TacticState, Dojo, ProofFinished, LeanError, ProofGivenUp, DojoHardTimeoutError
import utils
import textwrap

# ======================================================================================================================================
parser = argparse.ArgumentParser(description="Solve mathematical problems in Lean by ChatGPT with b-search")
parser.add_argument("--API_key", default=None, help="Openai API key",)
parser.add_argument("--model", default="gpt-4", help="GPT model version")
parser.add_argument("--temperature", default=0.7, type=float, help="Model Temperature")
parser.add_argument("--num_sample", default=4, type=int, help="Number of tactic samples")
parser.add_argument("--req_try", default=5, type=int, help="Number of retry for request of chatgpt")
parser.add_argument("--sleep_time", default=60, type=int, help="Time for intermediate sleep")

parser.add_argument("--data_path", default=None, help='Path for dataset')
parser.add_argument("--split", default='test', help='Name of test JSON file')
parser.add_argument("--ex_data", default=None, help='File path for example in prompt')

parser.add_argument("--file_path", default=None, help="File path containing theorem")
parser.add_argument("--full_name", default=None, help="Full name of theorem")
parser.add_argument("--name_filter", default=None, help="Name for filtering theorem")
parser.add_argument("--num_theorems", default=None, type=int, help="The number of theorems to load")
parser.add_argument("--timeout", default=600, type=int, help="Timeout for proof search")

parser.add_argument("--result_dir", default="results", help="Directory for searching result",)
parser.add_argument("--result_fname", default="minif2f_chatlean_bfs.jsonl", help="Name of result file")
parser.add_argument("--print_iter", default=10, type=int, help="Iteration number for print")

args = parser.parse_args()
logger.info(args)


# ======================================================================================================================================
def preparation(args):

    openai.api_key = args.API_key

    msg_dict = {}
    msg_dict["sys_message"] = "You are an expert in Lean3 theorem prover."
    msg_dict[
        "prompt"
    ] = textwrap.dedent("""\
    Make a proof statement in Lean3 to prove theorem using the following guidelines:

    - Generate only the single line of proof that immediately follows.
    - Do not use `sorry`.
    - All generated Unicode characters must retain a valid format for decoding.

    Here are some examples you may refer to:

    =========

    """)

    if args.ex_data is not None:
        examples = []
        with open(args.ex_data, "r", encoding="utf-8") as f:
            for line in f:
                examples.append(json.loads(line))

        for ex in examples:
            msg_dict[
                "prompt"
            ] += """Lean3 tactic state : \n{}\n\nNext tactic:\n\n%%%%%\n{}\n%%%%%\n\n=========\n\n""".format(
                ex["statement"], ex["tactic"]
            )

    repo, theorems, positions = utils._get_theorems(
        args.data_path,
        args.split,
        args.file_path,
        args.full_name,
        args.name_filter,
        args.num_theorems,
    )
    logger.info("The repository to test : {}".format(repo))

    return msg_dict, repo, theorems, positions


# ======================================================================================================================================
@dataclass
class Node:
    """
    tactic: str. proof in tactic form.

    status : str. one of the "Open", "Proved", "Failed".

    parent : Node. parent node list.

    ts : TacticState(pp=str, id=int, message=None). LeanDojo dataclass. Use in dojo.run_tac()

    children : List[Node, Node, ... Node] and its length = args.num_sample
    """

    tactic: str = None
    status: str = "Open"
    parent: list = None
    ts: TacticState = None 
    children: list = field(default_factory=list)


# ======================================================================================================================================
def generate(args, state, msg_dict):
    full_message = (
        msg_dict["prompt"]
        + textwrap.dedent("""\
            Then the next line is what we need to prove:

            Lean3 tactic state :
            {}

            Next tactic:

            """.format(
            state.pp
            )
        )
    )

    message = [
        {"role": "system", "content": msg_dict["sys_message"]},
        {"role": "user", "content": full_message},
    ]

    retries = args.req_try
    while retries > 0:
        try:
            logger.info("Generating")
            chatgpt_starttime = time.time()
            responses = openai.ChatCompletion.create(
                model=args.model,
                messages=message,
                temperature=args.temperature,
                n=args.num_sample,
            )
            res_time = time.time() - chatgpt_starttime
            break
        except ValueError:
            logger.warning(traceback.format_exc())
            logger.warning("error ValueError")
            raise ValueError
        except DojoHardTimeoutError:
            logger.warning(traceback.format_exc())
            logger.warning("error DojoHardTimeoutError")
            raise DojoHardTimeoutError
        except:  # noqa: E722
            logger.warning(traceback.format_exc())
            logger.warning(f"Retrying after {args.sleep_time} sec")
            retries -= 1
            time.sleep(args.sleep_time)

    tac_lst = []
    for one_response in responses["choices"]:
        res = one_response["message"]["content"].strip()
        tac_lst.append(
            res[res.find("%%%%%") + 6:res.find("%%%%%", res.find("%%%%%") + 1) - 1]
        )

    return (
        tac_lst,
        responses["usage"]["prompt_tokens"],
        responses["usage"]["completion_tokens"],
        res_time,
    )


def find_proved_node(children):
    for node in children:
        if node.status == "Proved":
            return node
    return None


def find_path(proved_node):
    reverse_path = [proved_node.tactic]
    current_node = proved_node
    while current_node.parent is not None:
        if current_node.parent.tactic is not None:
            reverse_path.append(current_node.parent.tactic)
        current_node = current_node.parent
    return list(reversed(reverse_path))


def write_result(
    args,
    tot_results,
    last=False,
):
    range_num = args.print_iter if not last else len(tot_results)

    with open("/".join([args.result_dir, args.result_fname]), "a") as h:
        for j in range(0, range_num):
            json.dump(tot_results[j], h)
            h.write("\n")


# ======================================================================================================================================


def run(args, theorems, msg_dict, repo):
    tot_start_time = time.time()
    tot_results = []
    result = {
        'repo': "/".join([repo.url, repo.commit]),
        'theorem_path': None,
        'theorem_name': None,
        'init_state': None,
    }

    for i, theorem in enumerate(theorems):
        logger.info(str(i + 1) + " " + str(theorem))
        result['theorem_path'] = theorem.file_path.__str__()
        result['theorem_name'] = theorem.full_name

        try:
            start_time2 = time.time()
            with Dojo(theorem, hard_timeout=args.timeout) as (dojo, init_state):
                result['init_state'] = init_state.pp
                start_time1 = time.time()
                accum_prompt_tokens = 0
                accum_generated_tokens = 0
                chatgpt_time = 0
                num_opennode, num_provednode = 1, 0
                child_ls = []
                num_node = []
                lastnode_ls, allpath = [], []

                # Start a recursive process.
                init_node = Node(None, "Open", None, init_state)
                generation = [init_node]

                while True:
                    next_generation = []
                    for node in generation:
                        # Judgment the status of node
                        if node.parent is not None:
                            try:
                                dojo_state = dojo.run_tac(node.parent.ts, node.tactic)
                                if isinstance(dojo_state, ProofFinished):
                                    logger.info("Theorem is proved")
                                    node.status = "Proved"
                                    num_provednode += 1
                                elif isinstance(dojo_state, LeanError):
                                    logger.info("Error in Lean is raised")
                                    node.status = "Failed: Error"
                                elif isinstance(dojo_state, TimeoutError):
                                    logger.info("Timed out")
                                    node.status = "Failed: Timeout"
                                elif isinstance(dojo_state, ProofGivenUp):
                                    logger.info("Proving is given up")
                                    node.status = "Failed: GiveUp"
                                else:
                                    logger.info("Continue proof")
                                    node.status = "Open"
                                    node.ts = dojo_state
                                    num_opennode += 1
                            except Exception as e:
                                logger.info(f"Error for implement: {i+1} {e}")
                                node.status = "Failed: Exception"
                                continue

                    num_node.append((num_opennode, num_provednode))
                    child_ls.append(len(generation))
                    num_opennode, num_provednode = 0, 0
                    for node in generation:
                        if node.status != "Open":
                            lastnode_ls.append(node)

                    # Decide whether to go to the next step or not.
                    all_failed = True
                    is_proved = False
                    for node in generation:
                        if node.status == "Proved":
                            is_proved = True
                            all_failed = False
                            break
                        elif node.status == "Open":
                            all_failed = False
                    if is_proved or all_failed:
                        break  # break while

                    if time.time() - start_time1 > args.timeout:
                        logger.warning("dojo error Timeout")
                        break  # break while due to timeout.

                    # Open node in the next step.
                    for node in generation:
                        if node.status == "Open":
                            # Generate the childern of Open status node
                            (
                                tac_lst,
                                prompt_tokens,
                                generated_tokens,
                                chatgpt_creating_time,
                            ) = generate(args, node.ts, msg_dict)

                            node.children = [
                                Node(cand_tactic, "Open", node)
                                for cand_tactic in list(set(tac_lst))
                            ]

                            next_generation += node.children

                            # results
                            accum_prompt_tokens += prompt_tokens
                            accum_generated_tokens += generated_tokens
                            chatgpt_time += chatgpt_creating_time

                    # update
                    generation = next_generation

        except:  # noqa: E722
            err_msg = traceback.format_exc()
            if "DojoHardTimeoutError" in err_msg:
                logger.warning("dojo error DojoHardTimeoutError")
            logger.warning(err_msg)

        end_time = time.time()
        searching_time1 = end_time - start_time1
        searching_time2 = end_time - start_time2

        if all_failed:
            proved_node = None
        else:
            proved_node = find_proved_node(generation)
        searching_time3 = time.time() - start_time2

        # find all path regardless of node.status
        for node in generation:
            if node.status == "Open":
                lastnode_ls.append(node)
        for node in lastnode_ls:
            path = find_path(node)
            allpath.append({"status": node.status, "path": path})

        # append results
        result['status'] = "Failed" if proved_node is None else "Proved"
        result['proof'] = [] if proved_node is None else find_path(proved_node)
        result['searching_time'] = (searching_time1, searching_time2, searching_time3)
        result['prompt_tokens'] = accum_prompt_tokens
        result['generated_tokens'] = accum_generated_tokens
        result['num_child'] = child_ls
        result['chatgpt_time'] = chatgpt_time
        result['node_open_proved'] = num_node
        result['all_path'] = allpath
        tot_results.append(result)

        # reset
        result = {
            'repo': "/".join([repo.url, repo.commit]),
            'theorem_path': None,
            'theorem_name': None,
            'init_state': None,
        }

        if len(tot_results) % args.print_iter == 0:
            logger.info(
                "{}th theorem is done. Total Time : {:.2f}".format(
                    i + 1, time.time() - tot_start_time
                )
            )

            write_result(
                args,
                tot_results
            )

            # initialize results
            tot_results = []

            logger.info("Results are saved until {}th theorem".format(i + 1))
            time.sleep(60)

    return (
        tot_results
    )


# ======================================================================================================================================
msg_dict, repo, theorems, positions = preparation(args)

if not os.path.isdir(args.result_dir):
    os.mkdir(f"{args.result_dir}")

(
    tot_results
) = run(args, theorems, msg_dict, repo)

if len(tot_results) != 0:
    last = True
    write_result(
        args,
        tot_results,
        last=last,
    )

logger.info("Result file is saved")
logger.info("Test over")
