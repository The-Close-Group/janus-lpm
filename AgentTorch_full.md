# AgentTorch: Complete Documentation
## This file contains all documentation from the AgentTorch project

## Document: architecture.md

# Framework Architecture

This document details the architecture of the AgentTorch project, explains all
the building blocks involved and points to relevant code implementation and
examples.

---

A high-level overview of the AgentTorch Python API is provided by the following
block diagram:

![Agent Torch Block Diagram](https://github.com/agenttorch/agenttorch/assets/34235681/3ecadf85-d949-44a6-92b3-5dbfa337fb20)

The AgentTorch Python API provides developers with the ability to
programmatically create and configure LPMs. This functionality is detailed
further in the following sections.

#### Runtime

The AgentTorch runtime is composed of three essential blocks: the configuration,
the registry, and the runner.

The configuration holds information about the environment, initial and current
state, agents, objects, network metadata, as well as substep definitions. The
'configurator' is defined in
[`config.py`](https://github.com/agenttorch/agenttorch/agent_torch/config.py).

The registry stores all registered substeps, and helper functions, to be called
by the runner. It is defined in
[`registry.py`](https://github.com/agenttorch/agenttorch/agent_torch/registry.py).

The runner accepts a registry and configuration, and exposes an API to execute
all, single or multiple episodes/steps in a simulation. It also maintains the
state and trajectory of the simulation across these episodes. It is defined in
[`runner.py`](https://github.com/agenttorch/agenttorch/agent_torch/runner.py),
and the substep execution and optimization logic is part of
[`controller.py`](https://github.com/agenttorch/agenttorch/agent_torch/controller.py).

#### Data

The data layer is composed of any raw, domain-specific data used by the model
(such as agent or object initialization data, environment variables, etc.) as
well as the files (YAML or Python code) used to configure the model. An example
of domain-specific data for a LPM can be found in the
[`models/covid/data`](/models/covid/data) folder. The configuration for the same
model can be found in [`config.yaml`](/models/covid/config.yaml).

#### Base Classes

The base classes of `Agent`, `Object` and `Substep` form the foundation of the
simulation. The agents defined in the configuration learn and interact with
either their environment, other agents, or objects through substeps. Substeps
are executed in the order of their definition in the configuration, and are
split into three parts:
[`SubstepObservation`](https://github.com/agenttorch/agenttorch/agent_torch/substep.py#L10),
[`SubstepAction`](https://github.com/agenttorch/agenttorch/agent_torch/substep.py#L27)
and
[`SubstepTransition`](https://github.com/agenttorch/agenttorch/agent_torch/substep.py#45).

- A `SubstepObservation` is defined to observe the state, and pick out those
  variables that are of use to the current substep.
- A `SubstepAction`, sometimes called a `SubstepPolicy`, decides the course of
  action based on the observations made, and then simulates that action.
- A `SubstepTransition` outputs the updates to be made to state variables based
  on the action taken in the substep.

An example of a substep will all three parts defined can be found
[here](/models/covid/substeps/quarantine).

#### Domain Extended Classes

These classes are defined by the developer/user configuring the model, in
accordance with the domain of the model. For example,
[in the COVID model](/models/covid), citizens of the populace
[are defined as `Agents`](/models/covid/config.yaml#L189), and `Transmission`
and `Quarantine` [as substeps](/models/covid/substeps).

---

## Document: install.md

# Installation Guide

> AgentTorch is meant to be used in a Python 3.9 environment (or above). If you have not
> installed Python 3.9, please do so first from
> [python.org/downloads](https://www.python.org/downloads/).

To install the project, run:

```sh
> pip install git+https://github.com/agenttorch/agenttorch
```

To run some models, you may need to separately install their dependencies. These
usually include [`torch`](https://pytorch.org/get-started/locally/),
[`torch_geometric`](https://github.com/pyg-team/pytorch_geometric#pytorch-20),
and [`osmnx`](https://osmnx.readthedocs.io/en/stable/installation.html).

For the sake of completeness, a summary of the commands required is given below:

```sh
# on macos, cuda is not available:
> pip install torch torchvision torchaudio
> pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv
> pip install osmnx

# on ubuntu, where ${CUDA} is the cuda version:
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA}
> pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
> pip install osmnx
```

## Hardware

The code has been tested on macOS Catalina 10.1.7 and Ubuntu 22.04.2 LTS.
Large-scale experiments are run using Nvidia's TitanX and V100 GPUs.

---

## Document: index.md

<h1 align="center">
  <a href="https://lpm.media.mit.edu/" target="_blank">
    Large Population Models
  </a>
</h1>

<p align="center">
  <strong>making complexity simple</strong><br>
  differentiable learning over millions of autonomous agents
</p>

<p align="center">
  <a href="https://agenttorch.github.io/AgentTorch/" target="_blank">
    <img src="https://img.shields.io/badge/Quick%20Introduction-green" alt="Documentation" />
  </a>
  <a href="https://twitter.com/intent/follow?screen_name=ayushchopra96" target="_blank">
    <img src="https://img.shields.io/twitter/follow/ayushchopra96?style=social&label=Get%20in%20Touch" alt="Get in Touch" />
  </a>
  <a href="https://join.slack.com/t/largepopulationmodels/shared_invite/zt-2jalzf9ki-n9nXG5FryVSMaPmEL7Wm2w" target="_blank">
     <img src="https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=white" alt="Join Us"/>
  </a>
</p>

## Overview

Many grand challenges like climate change and pandemics emerge from complex interactions of millions of individual decisions. While LLMs and AI agents excel at individual behavior, they can't model these intricate societal dynamics. Enter Large Population Models LPMs: a new AI paradigm simulating millions of interacting agents simultaneously, capturing collective behaviors at societal scale. It's like scaling up AI agents exponentially to understand the ripple effects of countless decisions.

AgentTorch, our open-source platform, makes building and running these massive simulations accessible. It's optimized for GPUs, allowing efficient simulation of entire cities or countries. Think PyTorch, but for large-scale agent-based simulations. AgentTorch LPMs have four design principles:

- **Scalability**: AgentTorch models can simulate country-size populations in
  seconds on commodity hardware.
- **Differentiability**: AgentTorch models can differentiate through simulations
  with stochastic dynamics and conditional interventions, enabling
  gradient-based learning.
- **Composition**: AgentTorch models can compose with deep neural networks (eg:
  LLMs), mechanistic simulators (eg: mitsuba) or other LPMs. This helps describe
  agent behavior using LLMs, calibrate simulation parameters and specify
  expressive interaction rules.
- **Generalization**: AgentTorch helps simulate diverse ecosystems - humans in
  geospatial worlds, cells in anatomical worlds, autonomous avatars in digital
  worlds.

LPMs are already making real-world impact. They're being used to help immunize millions of people by optimizing vaccine distribution strategies, and to track billions of dollars in global supply chains, improving efficiency and reducing waste. Our long-term goal is to "re-invent the census": built entirely in simulation, captured passively and used to protect country-scale populations. Our research is early but actively making an impact - winning awards at AI conferences and being deployed across the world. Learn more about LPMs [here](https://lpm.media.mit.edu/research.pdf).

AgentTorch is building the future of decision engines - inside the body, around us and beyond!

https://github.com/AgentTorch/AgentTorch/assets/13482350/4c3f9fa9-8bce-4ddb-907c-3ee4d62e7148

## Installation

The easiest way to install AgentTorch (v0.4.0) is from pypi:
```
> pip install agent-torch
```

> AgentTorch is meant to be used in a Python 3.9 environment. If you have not
> installed Python 3.9, please do so first from
> [python.org/downloads](https://www.python.org/downloads/).

Install the most recent version from source using `pip`:

```sh
> pip install git+https://github.com/agenttorch/agenttorch
```

> Some models require extra dependencies that have to be installed separately.
> For more information regarding this, as well as the hardware the project has
> been run on, please see [`docs/install.md`](docs/install.md).

## Getting Started

The following section depicts the usage of existing models and population data
to run simulations on your machine. It also acts as a showcase of the Agent
Torch API.

A Jupyter Notebook containing the below examples can be found
[here](docs/tutorials/using-models/walkthrough.ipynb).

### Executing a Simulation

```py
# re-use existing models and population data easily
from agent_torch.models import covid
from agent_torch.populations import astoria

# use the executor to plug-n-play
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation

# agent_"torch" works seamlessly with the pytorch API
from torch.optim import SGD

loader = LoadPopulation(astoria)
simulation = Executor(model=covid, pop_loader=loader)

simulation.init(SGD)
simulation.execute()
```

## Guides and Tutorials

### Understanding the Framework

A detailed explanation of the architecture of the Agent Torch framework can be
found [here](architecture.md).

### Creating a Model

A tutorial on how to create a simple predator-prey model can be found in the
[`tutorials/`]tutorials/) folder.

### Prompting Collective Behavior with LLM Archetypes

```py
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.behavior import Behavior
from agent_torch.core.llm.backend import LangchainLLM
from agent_torch.populations import NYC

user_prompt_template = "Your age is {age} {gender},{unemployment_rate} the number of COVID cases is {covid_cases}."

# Using Langchain to build LLM Agents
agent_profile = "You are a person living in NYC. Given some info about you and your surroundings, decide your willingness to work. Give answer as a single number between 0 and 1, only."
llm_langchian = LangchainLLM(
    openai_api_key=OPENAI_API_KEY, agent_profile=agent_profile, model="gpt-3.5-turbo"
)

# Create an object of the Archetype class
# n_arch is the number of archetypes to be created. This is used to calculate a distribution from which the outputs are then sampled.
archetype = Archetype(n_arch=7)

# Create an object of the Behavior class
# You have options to pass any of the above created llm objects to the behavior class
# Specify the region for which the behavior is to be generated. This should be the name of any of the regions available in the populations folder.
earning_behavior = Behavior(
    archetype=archetype.llm(llm=llm_langchian, user_prompt=user_prompt_template), region=NYC
)

kwargs = {
    "month": "January",
    "year": "2020",
    "covid_cases": 1200,
    "device": "cpu",
    "current_memory_dir": "/path-to-save-memory",
    "unemployment_rate": 0.05,
}

output = earning_behavior.sample(kwargs)
```

### Contributing to Agent Torch

Thank you for your interest in contributing! You can contribute by reporting and
fixing bugs in the framework or models, working on new features for the
framework, creating new models, or by writing documentation for the project.

Take a look at the [contributing guide](contributing.md) for instructions
on how to setup your environment, make changes to the codebase, and contribute
them back to the project.

## Impact

> **AgentTorch models are being deployed across the globe.**

![Impact](media/impact.png)

---

## Document: contributing.md

# Contributing Guide

Thanks for your interest in contributing to Agent Torch! This guide will show
you how to set up your environment and contribute to this library.

## Prerequisites

You must have the following software installed:

1. [`git`](https://github.com/git-guides/install-git) (latest)
2. [`python`](https://wiki.python.org/moin/BeginnersGuide/Download) (>= 3.10)

Once you have installed the above, follow
[these instructions](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
to
[`fork`](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks)
and [`clone`](https://github.com/git-guides/git-clone) the repository
(`AgentTorch/AgentTorch`).

Once you have forked and cloned the repository, you can
[pick out an issue](https://github.com/AgentTorch/AgentTorch/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc)
you want to fix/implement!

## Making Changes

Once you have cloned the repository to your computer (say, in
`~/Code/AgentTorch`) and picked the issue you want to tackle, create a virtual
environment, install all dependencies, and create a new branch to hold all your
work.

```sh
# create a virtual environment
> python -m venv .venv/

# set it up
> . .venv/bin/activate
> pip install -r development.txt
> pip install -e .

# set up the pre commit hooks
> pre-commit install --config pre-commit.yaml

# create a new branch
> git switch master
> git switch --create branch-name
```

While naming your branch, make sure the name is short and self explanatory.

Once you have created a branch, you can start coding!

## Project Structure

The project is structured as follows. The comments written next to the
file/folder give a brief explanation of what purpose the file/folder serves.

```sh
.
├── agent_torch/
│  ├── helpers/ # defines helper functions used to initialize or work with the state of the simulation.
│  ├── llm/ # contains all the code related to using llms as agents in the simulation
│  ├── __init__.py # exports everything to the world
│  ├── config.py # handles reading and processing the simulation's configuration
│  ├── controller.py # executes the substeps for each episode
│  ├── initializer.py # creates a simulation from a configuration and registry
│  ├── registry.py # registry that stores references to the implementations of the substeps and helper functions
│  ├── runner.py # executes the episodes of the simulation, and handles its state
│  ├── substep.py # contains base classes for the substep observations, actions and transitions
│  └── utils.py # utility functions used throughout the project
├── docs/
│  ├── media/ # assets like screenshots or diagrams inserted in .md files
│  ├── tutorials/ # jupyter notebooks with tutorials and their explanations
│  ├── architecture.md # the framework's architecture
│  └── install.md # instructions on installing the framework
├── models/
│  ├── covid/ # a model simulating disease spread, using the example of covid 19
│  └── predator_prey/ # a simple model used to showcase the features of the framework
├── citation.bib # contains the latex code to use to cite this project
├── contributing.md # this file, helps onboard contributors
├── license.md # contains the license for this project (MIT)
├── readme.md # contains details on the what, why, and how
├── requirements.txt # lists the dependencies of the framework
└── setup.py # defines metadata for the project
```

Note that after making any code changes, you should run the `black` code
formatter, as follows:

```sh
> black agent_torch/ tests/
```

You should also ensure all the unit tests pass, especially if you have made
changes to any files in the `agent_torch/` folder.

```sh
> pytest -vvv tests/
```

For any changes to the documentation, run `prettier` over the `*.md` files after
making changes to them. To preview the generated documentation, run:

```sh
> mkdocs serve
```

> Rememeber to add any new pages to the sidebar by editing `mkdocs.yaml`.

If you wish to write a tutorial, write it in a Jupyter Notebook, and then
convert it to a markdown file using `nbconvert`:

```sh
> pip install nbconvert
> jupyter nbconvert --to markdown <file>.ipynb
> mv <file>.md index.md
```

> Rememeber to move any files that it generates to the `docs/media` folder, and
> update the hyperlinks in the generated markdown file.

## Saving Changes

After you have made changes to the code, you will want to
[`commit`](https://github.com/git-guides/git-commit) (basically, Git's version
of save) the changes. To commit the changes you have made locally:

```sh
> git add this/folder that-file.js
> git commit --message 'commit-message'
```

While writing the `commit-message`, try to follow the below guidelines:

Prefix the message with `type:`, where `type` is one of the following
dependending on what the commit does:

- `fix`: Introduces a bug fix.
- `feat`: Adds a new feature.
- `test`: Any change related to tests.
- `perf`: Any performance related change.
- `meta`: Any change related to the build process, workflows, issue templates,
  etc.
- `refc`: Any refactoring work.
- `docs`: Any documentation related changes.

Try to keep the first line brief, and less than 60 characters. Describe the
change in detail in a new paragraph (double newline after the first line).

## Contributing Changes

Once you have committed your changes, you will want to
[`push`](https://github.com/git-guides/git-push) (basically, publish your
changes to GitHub) your commits. To push your changes to your fork:

```sh
> git push origin branch-name
```

If there are changes made to the `master` branch of the `AgentTorch/AgentTorch`
repository, you may wish to merge those changes into your branch. To do so, you
can run the following commands:

```
> git fetch upstream master
> git merge upstream/master
```

This will automatically add the changes from `master` branch of the
`AgentTorch/AgentTorch` repository to the current branch. If you encounter any
merge conflicts, follow
[this guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-using-the-command-line)
to resolve them.

Once you have pushed your changes to your fork, follow
[these instructions](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)
to open a
[`pull request`](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests):

Once you have submitted a pull request, the maintainers of the repository will
review your pull requests and provide feedback. If they find the work to be
satisfactory, they will merge the pull request.

#### Thanks for contributing!

<!-- This contributing guide was inspired by the Electron project's contributing guide. -->

---

## Document: tutorials/calibrating-a-model/index.md

# Calibrating an AgentTorch Model

This tutorial demonstrates how to calibrate parameters in an AgentTorch model using different optimization approaches. We'll explore three methods for parameter optimization and discuss when to use each approach.

## Prerequisites

- Basic understanding of PyTorch and gradient-based optimization
- Familiarity with AgentTorch's basic concepts
- Python 3.8+
- Required packages listed in `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

## Overview

Model calibration is a crucial step in agent-based modeling. AgentTorch provides several approaches to optimize model parameters:

1. Internal parameter optimization
2. External parameter optimization
3. Generator-based parameter optimization

## Basic Setup

First, let's set up our environment and import the necessary modules:

```python
import warnings
warnings.simplefilter("ignore")

import torch
import torch.nn as nn
from agent_torch.models import covid
from agent_torch.populations import sample
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation

# Initialize simulation
sim = Executor(covid, pop_loader=LoadPopulation(sample))
runner = sim.runner
runner.init()
```

## Helper Classes and Functions

We'll define some helper components that we'll use throughout the tutorial:

```python
class LearnableParams(nn.Module):
    """A neural network module that generates bounded parameters"""
    def __init__(self, num_params, device='cpu'):
        super().__init__()
        self.device = device
        self.num_params = num_params
        self.learnable_params = nn.Parameter(torch.rand(num_params, device=self.device))
        self.min_values = torch.tensor(2.0, device=self.device)
        self.max_values = torch.tensor(3.5, device=self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        out = self.learnable_params
        # Bound output between min_values and max_values
        out = self.min_values + (self.max_values - self.min_values) * self.sigmoid(out)
        return out

def execute(runner, n_steps=5):
    """Execute simulation and compute loss"""
    runner.step(n_steps)
    labels = runner.state_trajectory[-1][-1]['environment']['daily_infected']
    return labels.sum()
```

## Method 1: Internal Parameter Optimization

The first approach optimizes parameters that are internal to the simulation. This is useful when you want to directly optimize parameters that are part of your model's structure.

```python
def optimize_internal_params():
    # Execute simulation and compute gradients
    loss = execute(runner)
    loss.backward()
    
    # Get gradients of learnable parameters
    learn_params_grad = [(name, param, param.grad) 
                        for (name, param) in runner.named_parameters()]
    return learn_params_grad

# Example usage
gradients = optimize_internal_params()
print("Internal parameter gradients:", gradients)
```

### When to use this method?
- When parameters are naturally part of your simulation structure
- When you want direct control over parameter optimization
- For simpler models with fewer parameters

## Method 2: External Parameter Optimization

The second approach involves optimizing external parameters that are fed into the simulation. This provides more flexibility in parameter management.

```python
def optimize_external_params():
    # Create external parameters
    external_params = nn.Parameter(
        torch.tensor([2.7, 3.8, 4.6], requires_grad=True)[:, None]
    )
    
    # Set parameters in the runner
    learnable_params = runner.named_parameters()
    params_dict = {next(iter(learnable_params))[0]: external_params}
    runner._set_parameters(params_dict)
    
    # Execute and compute gradients
    loss = execute(runner)
    loss.backward()
    return external_params.grad

# Example usage
gradients = optimize_external_params()
print("External parameter gradients:", gradients)
```

### When to use this method?
- When you want to manage parameters outside the simulation
- For parameter sweeps or sensitivity analysis
- When parameters need to be shared across different components

## Method 3: Generator-Based Parameter Optimization

The third approach uses a generator function to predict optimal parameters. This is particularly useful for complex parameter relationships.

```python
def optimize_with_generator():
    # Create generator model
    learn_model = LearnableParams(3)
    params = learn_model()[:, None]
    
    # Execute and compute gradients
    loss = execute(runner)
    loss.backward()
    
    # Get gradients of generator parameters
    learn_params_grad = [(param, param.grad) 
                        for (name, param) in learn_model.named_parameters()]
    return learn_params_grad

# Example usage
gradients = optimize_with_generator()
print("Generator parameter gradients:", gradients)
```

### When to use this method?
- When parameters have complex relationships
- For learning parameter patterns
- When you want to generate parameters based on conditions

## Putting It All Together

Here's how to use all three methods in a complete optimization loop:

```python
def calibrate_model(method='internal', num_epochs=10):
    for epoch in range(num_epochs):
        if method == 'internal':
            gradients = optimize_internal_params()
        elif method == 'external':
            gradients = optimize_external_params()
        else:  # generator
            gradients = optimize_with_generator()
            
        print(f"Epoch {epoch}, gradients: {gradients}")
        # Add your optimizer step here

# Example usage
calibrate_model(method='internal', num_epochs=3)
```

## Best Practices

1. **Choose the Right Method**: Consider your specific use case when selecting an optimization approach.
2. **Monitor Convergence**: Always track your loss function to ensure proper optimization.
3. **Validate Results**: Cross-validate your calibrated parameters with held-out data.
4. **Handle Constraints**: Use appropriate bounds and constraints for your parameters.

## Common Pitfalls

- Ensure parameters have appropriate ranges
- Watch out for local optima
- Be careful with learning rates in optimization
- Consider the computational cost of each approach

## Conclusion

We've explored three different approaches to model calibration in AgentTorch. Each method has its strengths and is suited for different scenarios. Choose the approach that best matches your specific needs and model complexity.

## Additional Resources

- [AgentTorch Documentation](https://agent-torch.ai/)
- [PyTorch Optimization](https://pytorch.org/docs/stable/optim.html)
- [Related Tutorials](../index.md)

---

## Document: tutorials/configure-behavior/index.md

# Guide to Prompting LLM as ABM Agents

Welcome to this comprehensive tutorial on AI agent behavior generation! This
guide is designed for newcomers who want to learn how to create AI agents and
simulate population behaviors using a custom framework. We'll walk you through
each step, explaining concepts as we go.

## Table of Contents

1. [Introduction to the Framework](#introduction)
2. [Setting Up Your Environment](#setup)
3. [Understanding the Core Components](#components)
4. [Creating Your First AI Agent](#first-agent)
5. [Generating Population Behaviors](#behaviors)
6. [Putting It All Together](#all-together)
7. [Next Steps and Advanced Topics](#next-steps)

<a name="introduction"></a>

## 1. Introduction to the Framework

Our framework is designed to simulate population behaviors using AI agents. It
combines several key components:

- **LLM Agents**: We use Large Language Models (LLMs) to create intelligent
  agents that can make decisions based on given scenarios.
- **Archetypes**: These represent different types of individuals in a
  population.
- **Behaviors**: These simulate how individuals might act in various situations.

This framework is particularly useful for modeling complex social or economic
scenarios, such as population responses during a pandemic.

<a name="setup"></a>

## 2. Setting Up Your Environment

Now, let's set up your OpenAI API key (you'll need an OpenAI account):

```python
OPENAI_API_KEY = None # Replace with your actual API key
```

<a name="components"></a>

## 3. Understanding the Core Components

Let's break down the main components of our framework:

### DspyLLM and LangchainLLM

These are wrappers around language models that allow us to create AI agents.
They can process prompts and generate responses based on given scenarios.

### Archetype

This component helps create different "types" of individuals in our simulated
population. Like Male under 19, Female from 20 to 29 years of age.

### Behavior

The Behavior component simulates how individuals (or groups) might act in
various situations. It uses the AI agents to generate these behaviors.

Now you have two AI agents ready to process prompts!

## 4. Creating LLM Agents

We support using Langchain and Dspy backends to initialize LLM instances - for agent and archetypes. Using our [LLMBackend class](https://github.com/AgentTorch/AgentTorch/blob/1b2a723aa8da4d47f30870af51e87a55cea838b8/agent_torch/core/llm/backend.py#L17), you can integrate any framework of your choice.

### Using DSPy

```python
from dspy_modules import COT, BasicQAWillToWork
from agent_torch.core.llm.llm import DspyLLM

llm_dspy = DspyLLM(qa=BasicQAWillToWork, cot=COT, openai_api_key=OPENAI_API_KEY)
llm_dspy.initialize_llm()

output_dspy = llm_dspy.prompt(["You are an individual living during the COVID-19 pandemic. You need to decide your willingness to work each month and portion of your assets you are willing to spend to meet your consumption demands, based on the current situation of NYC."])
print("DSPy Output:", output_dspy)
```

### Using Langchain

```python
from agent_torch.core.llm.llm import LangchainLLM

agent_profile = "You are an helpful agent who is trying to help the user make a decision. Give answer as a single number between 0 and 1, only."

llm_langchian = LangchainLLM(openai_api_key=OPENAI_API_KEY, agent_profile=agent_profile, model="gpt-3.5-turbo")
llm_langchian.initialize_llm()

output_langchain = llm_langchian.prompt(["You are an helpful agent who is trying to help the user make a decision. Give answer as a single number between 0.0 and 1.0, only."])
print("Langchain Output:", output_langchain)
```

<a name="behaviors"></a>

## 5. Generating Population Behaviors

To simulate population behaviors, we'll use the Archetype and Behavior classes:

```python
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.behavior import Behavior
from agent_torch.populations import NYC

# Create an object of the Archetype class
# n_arch is the number of archetypes to be created. This is used to calculate a distribution from which the outputs are then sampled.
archetype = Archetype(n_arch=7)

# Define a prompt template
# Age,Gender and other attributes which are part of the population data, will be replaced by the actual values of specified region, during the simulation.
# Other variables like Unemployment Rate and COVID cases should be passed as kwargs to the behavior model.
user_prompt_template = "Your age is {age} {gender}, unemployment rate is {unemployment_rate}, and the number of COVID cases is {covid_cases}.Current month is {month} and year is {year}."

# Create a behavior model
# You have options to pass any of the above created llm objects to the behavior class
# Specify the region for which the behavior is to be sampled. This should be the name of any of the regions available in the populations folder.
earning_behavior = Behavior(
    archetype=archetype.llm(llm=llm_dspy, user_prompt=user_prompt_template, num_agents=12),
    region=NYC
)

print("Behavior model created successfully!")
```

This sets up a behavior model that can simulate how 12 different agents might
behave in NYC during the COVID-19 pandemic.

<a name="all-together"></a>

## 6. Putting It All Together

Now, let's use our behavior model to generate some population behaviors:

```python
# Define scenario parameters
# Names of the parameters should match the placeholders in the user_prompt template
scenario_params = {
    'month': 'January',
    'year': '2020',
    'covid_cases': 1200,
    'device': 'cpu',
    'current_memory_dir': '/path-to-save-memory',
    'unemployment_rate': 0.05,
}

# Generate behaviors
population_behaviors = earning_behavior.sample(scenario_params)
print("Population Behaviors:")
print(population_behaviors)
```

```python
# Define another scenario parameters
scenario_params = {
    'month': 'February',
    'year': '2020',
    'covid_cases': 900,
    'device': 'cpu',
    'current_memory_dir': '/path-to-save-memory',
    'unemployment_rate': 0.1,
}

# Generate behaviors
population_behaviors = earning_behavior.sample(scenario_params)
print("Population Behaviors:")
print(population_behaviors)
```

```python
# Define yet another scenario parameters
scenario_params = {
    'month': 'March',
    'year': '2020',
    'covid_cases': 200,
    'device': 'cpu',
    'current_memory_dir': '/path-to-save-memory',
    'unemployment_rate': 0.11,
}

# Generate behaviors
population_behaviors = earning_behavior.sample(scenario_params)
print("Population Behaviors:")
print(population_behaviors)
```

And so on...

This will output a set of behaviors for our simulated population based on the
given scenario.

<a name="next-steps"></a>

## 7. Next Steps and Advanced Topics

You've just created your first AI agents and simulated population behaviors.
Here are some advanced topics you might want to explore next:

- Customizing archetypes for specific populations
- Creating more complex behavior models

---

## Document: tutorials/processing-a-population/index.md

# Tutorial: Generating Base Population and Household Data

This tutorial will guide you through the process of generating base population
and household data for a specified region using census data. We’ll use a
`CensusDataLoader` class to handle the data processing and generation.

## Before Starting

Make sure your `population data` and `household data` are in the prescribed
format. Names of the column need to be same as shown in the excerpts.

Lets see a snapshot of the data

`Population Data` is a dictionary containing two pandas DataFrames:
'`age_gender`' and '`ethnicity`'. Each DataFrame provides demographic
information for different areas and regions.

The `age_gender` DataFrame provides a comprehensive breakdown of population
data, categorized by area, gender, and age group.

#### Columns Description

- `area`: Serves as a unique identifier for each geographical area, represented
  by a string (e.g., `'BK0101'`, `'SI9593'`).
- `gender`: Indicates the gender of the population segment, with possible values
  being `'female'` or `'male'`.
- `age`: Specifies the age group of the population segment, using a string
  format such as `'20t29'` for ages 20 to 29, and `'U19'` for those under 19
  years of age.
- `count`: Represents the total number of individuals within the specified
  gender and age group for a given area.
- `region`: A two-letter code that identifies the broader region encompassing
  the area (e.g., `'BK'` for Brooklyn, `'SI'` for Staten Island).

##### Example Entry

Here is a sample of the data structure within the `age_gender` DataFrame:

| area   | gender | age   | count | region |
| ------ | ------ | ----- | ----- | ------ |
| BK0101 | female | 20t29 | 3396  | BK     |
| BK0101 | male   | 20t29 | 3327  | BK     |

This example entry demonstrates the DataFrame's layout and the type of
demographic data it contains, highlighting its utility for detailed population
studies by age and gender.

The `ethnicity` DataFrame is structured to provide detailed population data,
segmented by both geographical areas and ethnic groups.

##### Columns Description

- `area`: A unique identifier assigned to each area, formatted as a string
  (e.g., `'BK0101'`, `'SI9593'`). This identifier helps in pinpointing specific
  locations within the dataset.
- `ethnicity`: Represents the ethnic group of the population in the specified
  area.
- `count`: Indicates the number of individuals belonging to the specified ethnic
  group within the area. This is an integer value representing the population
  count.
- `region`: A two-letter code that signifies the broader region that the area
  belongs to (e.g., `'BK'` for Brooklyn, `'SI'` for Staten Island).

##### Example Entry

Below is an example of how the data is presented within the DataFrame:

| area   | ethnicity | count | region |
| ------ | --------- | ----- | ------ |
| BK0101 | asian     | 1464  | BK     |
| BK0101 | black     | 937   | BK     |

This example illustrates the structure and type of data contained within the
`ethnicity` DataFrame, showcasing its potential for detailed demographic
studies.

`Household Data` contains the following columns:

- `area`: Represents a unique identifier for each area.
- `people_num`: The total number of people within the area.
- `children_num`: The number of children in the area.
- `household_num`: The total number of households.
- `family_households`: Indicates the number of households identified as family
  households, highlighting family-based living arrangements.
- `nonfamily_households`: Represents the number of households that do not fall
  under the family households category, including single occupancy and unrelated
  individuals living together.
- `average_household_size`: The average number of individuals per household.

Below is a sample excerpt:

| area   | people_num | children_num | household_num | family_households | nonfamily_households | average_household_size |
| ------ | ---------- | ------------ | ------------- | ----------------- | -------------------- | ---------------------- |
| 100100 | 104        | 56           | 418           | 1                 | 0                    | 2.488038               |
| 100200 | 132        | 73           | 549           | 1                 | 0                    | 2.404372               |
| 100300 | 5          | 0            | 10            | 0                 | 1                    | 5.000000               |

Now that we have verified our input, we can proceed to next steps!

## Step 1: Set Up File Paths

First, we need to specify the paths to our data files.

Make sure to replace the placeholder paths with the actual paths to your data
files.

```python
# Path to the population data file. Update with the actual file path.
POPULATION_DATA_PATH = "docs/tutorials/processing-a-population/sample_data/NYC/population.pkl"

# Path to the household data file. Update with the actual file path.
HOUSEHOLD_DATA_PATH = "docs/tutorials/processing-a-population/sample_data/NYC/household.pkl"
```

## Step 2: Define Age Group Mapping

We’ll define a mapping for age groups to categorize adults and children in the
household data:

```python
AGE_GROUP_MAPPING = {
    "adult_list": ["20t29", "30t39", "40t49", "50t64", "65A"],  # Age ranges for adults
    "children_list": ["U19"],  # Age range for children
}
```

## Step 3: Load Data

Now, let’s load the population and household data:

```python
import numpy as np
import pandas as pd

# Load household data
HOUSEHOLD_DATA = pd.read_pickle(HOUSEHOLD_DATA_PATH)

# Load population data
BASE_POPULATION_DATA = pd.read_pickle(POPULATION_DATA_PATH)
```

## Step 4: Set Up Additional Parameters

We’ll set up some additional parameters that might be needed for data
processing. These are not essential for generating population, but still good to
know if you decide to use them in future.

```python
# Placeholder for area selection criteria, if any. Update or use as needed.
# Example: area_selector = ["area1", "area2"]
# This will be used to filter the population data to only include the selected areas.
area_selector = None

# Placeholder for geographic mapping data, if any. Update or use as needed.
geo_mapping = None
```

## Step 5: Initialize the Census Data Loader

Create an instance of the `CensusDataLoader` class:

```python
from agent_torch.data.census.census_loader import CensusDataLoader

census_data_loader = CensusDataLoader(n_cpu=8, use_parallel=True)
```

This initializes the loader with 8 CPUs and enables parallel processing for
faster data generation.

## Step 6: Generate Base Population Data

Generate the base population data for a specified region:

```python
census_data_loader.generate_basepop(
    input_data=BASE_POPULATION_DATA,  # The population data frame
    region="astoria",  # The target region for generating base population
    area_selector=area_selector,  # Area selection criteria, if applicable
)
```

This will create a base population of 100 individuals for the “astoria” region.
The generated data will be exported to a folder named “astoria” under the
“populations” folder.

#### Overview of the Generated Base Population Data

Each row corresponds to attributes of individual residing in the specified
region while generating the population.

| area   | age   | gender | ethnicity | region |
| ------ | ----- | ------ | --------- | ------ |
| BK0101 | 20t29 | female | black     | BK     |
| BK0101 | 20t29 | female | hispanic  | BK     |
| ...    | ...   | ...    | ...       | ...    |
| BK0101 | U19   | male   | asian     | SI     |
| BK0101 | U19   | female | white     | SI     |
| BK0101 | U19   | male   | asian     | SI     |

## Step 7: Generate Household Data

Finally, generate the household data for the specified region:

```python
census_data_loader.generate_household(
    household_data=HOUSEHOLD_DATA,  # The loaded household data
    household_mapping=AGE_GROUP_MAPPING,  # Mapping of age groups for household composition
    region="astoria"  # The target region for generating households
)
```

This will create household data for the “astoria” region based on the previously
generated base population. The generated data will be exported to the same
“astoria” folder under the “populations” folder.

## Bonus: Generate Population Data of Specific Size

For quick experimentation, this may come in handy.

```python
census_data_loader.generate_basepop(
    input_data=BASE_POPULATION_DATA,  # The population data frame
    region="astoria",  # The target region for generating base population
    area_selector=area_selector,  # Area selection criteria, if applicable
    num_individuals = 100 # Saves data for first 100 individuals, from the generated population
)
```

## Bonus: Export Population Data

If you have already generated your synthetic population, you just need to export
it to "populations" folder under the desired "region", in order for you to use
it with AgentTorch.

```python
POPULATION_DATA_PATH = "/population_data.pickle"  # Replace with actual path
census_data_loader.export(population_data_path=POPULATION_DATA_PATH,region="astoria")
```

In case you want to export data for only few individuals

```python
census_data_loader.export(population_data_path=POPULATION_DATA_PATH,region="astoria",num_individuals = 100)
```

## Conclusion

You have now successfully generated both base population and household data for
the `“astoria”` region. The generated data can be found in the
`“populations/astoria”` folder. You can modify the region name, population size,
and other parameters to generate data for different scenarios.

---

## Document: tutorials/index.md

# Tutorials

The following tutorials, in alphabetical order, can be found in this folder:

- [Creating a simulation](creating-a-model/index.md)
- [Inserting a population](processing-a-population/index.md)
- [Gradient-based calibration](calibrating-a-model/index.md)
- [Prompting LLM as LPM Agents](configure-behavior/index.md)
- [From simulation to reality](integrating-with-beckn/index.md)

---

## Document: tutorials/compare-llm-performance/index.md

## Exploring LLM Influence

#### Introduction
AgentTorch is a framework that scales ABM simulations to real-world problems. This tutorial will guide you through the process of experimenting with different LLMs in AgentTorch to understand their impact on the framework's effectiveness.

#### Step 1: Setup
First, let's set up our environment and import the necessary libraries:


```python
import sys
from dspy_modules import COT, BasicQAWillToWork
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.behavior import Behavior
from agent_torch.populations import NYC
from agent_torch.core.llm.backend import DspyLLM, LangchainLLM

import torch
OPENAI_API_KEY = None
```

Setup : Covid Cases Data and Unemployment Rate


```python
from utils import get_covid_cases_data
csv_path = 'agent_torch/models/covid/data/county_data.csv'
monthly_cases_kings = get_covid_cases_data(csv_path=csv_path,county_name='Kings County')
monthly_cases_queens = get_covid_cases_data(csv_path=csv_path,county_name='Queens County')
monthly_cases_bronx = get_covid_cases_data(csv_path=csv_path,county_name='Bronx County')
monthly_cases_new_york = get_covid_cases_data(csv_path=csv_path,county_name='New York County')
monthly_cases_richmond = get_covid_cases_data(csv_path=csv_path,county_name='Richmond County')

```

#### Step 2: Initialise LLM Instance

We can use either of the Langchain and Dspy backends to initialise a LLM instance. While these are the frameworks we are supporting currently, you may choose to use your own framework of choice by extending the LLMBackend class provided with AgentTorch.

Let's see how we can use Langchain to initialise an LLM instance

GPT 3.5 Turbo


```python
agent_profile = "You are an helpful agent who is trying to help the user make a decision. Give answer as a single number between 0 and 1, only."
llm_langchain_35 = LangchainLLM(
    openai_api_key=OPENAI_API_KEY, agent_profile=agent_profile, model="gpt-3.5-turbo"
)
```

GPT 4-0 Mini


```python
agent_profile = "You are an helpful agent who is trying to help the user make a decision. Give answer as a single number between 0 and 1, only."
llm_langchain_4o_mini = LangchainLLM(
    openai_api_key=OPENAI_API_KEY, agent_profile=agent_profile, model="gpt-4o-mini"
) 
```

Similarly if we wanted to use Dspy backend, we can instantiate the DspyLLM object.
We can pass the desired model name as argument just like we did with Langchain.


```python
# Agent profile is decided by the QA module and the COT module enforces Chain of Thought reasoning
llm_dspy = DspyLLM(qa=BasicQAWillToWork, cot=COT, openai_api_key=OPENAI_API_KEY)
```

#### Step 3: Define agent Behavior

Create an object of the Behavior class
You have options to pass any of the above created llm objects to the behavior class
Specify the region for which the behavior is to be generated. This should be the name of any of the regions available in the populations folder.


```python
# Create an object of the Archetype class
# n_arch is the number of archetypes to be created. This is used to calculate a distribution from which the outputs are then sampled.
archetype = Archetype(n_arch=2) 

# Define a prompt template
# Age,Gender and other attributes which are part of the population data, will be replaced by the actual values of specified region, during the simulation.
# Other variables like Unemployment Rate and COVID cases should be passed as kwargs to the behavior model.
user_prompt_template = "Your age is {age}, gender is {gender}, ethnicity is {ethnicity}, and the number of COVID cases is {covid_cases}.Current month is {month} and year is {year}."

# Create a behavior model
# You have options to pass any of the above created llm objects to the behavior class
# Specify the region for which the behavior is to be sampled. This should be the name of any of the regions available in the populations folder.
earning_behavior_4o_mini = Behavior(
    archetype=archetype.llm(llm=llm_langchain_4o_mini, user_prompt=user_prompt_template),
    region=NYC
)
earning_behavior_35 = Behavior(
    archetype=archetype.llm(llm=llm_langchain_35, user_prompt=user_prompt_template),
    region=NYC
)
```


```python
# Define arguments to be used for creating a query for the LLM instance
kwargs = {
    "month": "January",
    "year": "2020",
    "covid_cases": 1200,
    "device": "cpu",
    "current_memory_dir": "/populations/astoria/conversation_history",
    "unemployment_rate": 0.05,
}
```

#### Step 4: Compare performance between different LLM models


```python
from utils import get_labor_data, get_labor_force_correlation

labor_force_df_4o_mini, observed_labor_force_4o_mini, correlation_4o_mini = get_labor_force_correlation(
    monthly_cases_kings, 
    earning_behavior_4o_mini, 
    'agent_torch/models/macro_economics/data/unemployment_rate_csvs/Brooklyn-Table.csv',
    kwargs
)
labor_force_df_35, observed_labor_force_35, correlation_35 = get_labor_force_correlation(
    monthly_cases_kings, 
    earning_behavior_35, 
    'agent_torch/models/macro_economics/data/unemployment_rate_csvs/Brooklyn-Table.csv',
    kwargs
)
print(f"Correlation with GPT 3.5 is {correlation_35} and with GPT 4o Mini is {correlation_4o_mini}")
```

---

## Document: tutorials/creating-a-model/index.md

# Predator-Prey Model

<details>
  <summary>Imports</summary>

```python
# import agent-torch

import os
import sys
module_path = os.path.abspath(os.path.join('../../../agent_torch'))
if module_path not in sys.path:
    sys.path.append(module_path)

from AgentTorch import Runner, Registry
from AgentTorch.substep import SubstepObservation, SubstepAction, SubstepTransition
from AgentTorch.helpers import get_by_path, read_config, read_from_file, grid_network
```

```python
# import all external libraries that we need.

import math
import torch
import re
import random
import argparse
import numpy as np
import torch.nn as nn
import networkx as nx
import osmnx as ox
from tqdm import trange
```

```python
# define the helper functions we need.

def get_var(state, var):
  """
    Retrieves a value from the current state of the model.
  """
  return get_by_path(state, re.split('/', var))
```

</details>

> The complete code for this model can be found
> [here](https://github.com/agenttorch/agenttorch/blob/master/models/predator_prey).
> The architecture of the AgentTorch framework, which explains some key
> concepts, can be found [here](../../architecture.md).

This guide walks you through creating a custom predator-prey model using the
AgentTorch framework. This model will simulate an ecosystem consisting of
predators, prey and grass: predators eat prey, and prey eat grass.

The model's parameters, rules and configuration are passed to AgentTorch, which
iteratively simulates the model, allowing you to optimize its learnable
parameters, while also modelling the simulation in real time. AgentTorch's
Python API is based on PyTorch, which enhances its performance on GPUs.

The following sections detail:

- an overview of the model's rules and parameters.
- the properties of all entities stored in the model's state.
- the substeps that observe, simulate and modify the state for each agent.
- the code required to run the simulation using `agent-torch`.
- plotting the state's trajectory using `matplotlib`.

## Model Overview

The following are configurable parameters of the model:

- a $n \times m$ grid, with $p$ predators and $q$ prey to start with.
- grass can grown on any of the $n \cdot m$ squares in the grid.

The rules followed by the simulated interactions are configured as follows:

- predators can eat only prey, and prey can eat only grass.
- grass grows back once eaten after a certain number of steps.
- upon consuming food, the energy of the consumer increases.
- movement happens randomly, to any neighbouring square in the grid.
- each move reduces the energy of the entity by a fixed amount.

These parameters and rules, along with the properties of the entities (detailed
below) in the simulation are defined in a configuration file, and passed on to
the model.

## State: Environment, Agents, and Objects

The model's state consists of a list of properties of the simulated environment,
and the agents and objects situated in that simulation. For this model, the:

### Environment

The environment will have only one property: the size of the two-dimensional
grid in which the predators and prey wander, defined like so:

```yaml
environment:
  bounds: (max_x, max_y) # tuple of integers
```

### Agents

This model has two agents: predator, and prey.

#### Predator

The predator agent is defined like so:

```yaml
predator:
  coordinates: (x, y) # tuple of integers
  energy: float
  stride_work: float
```

The `coordinates` property depicts the current position of the predator in the
two-dimensional grid. It is initialized from a CSV file that contains a list of
randomly generated coordinates for all 40 predators.

The `energy` property stores the current amount of energy possessed by the
predator. Initially, this property is set to a random number between 30 and 100.

The `stride_work` property is a static, but learnable property that stores the
amount of energy to deduct from a predator for one step in any direction on the
grid.

#### Prey

The prey agent is identical to the predator agent, and has one additional
property: `nutritional_value`.

```yaml
prey:
  coordinates: (x, y) # tuple of integers
  energy: float
  stride_work: float
  nutritional_value: float
```

The `nutritional_value` property is a static but learnable property that stores
the amount of energy gained by a predator when it consumes a single prey entity.

### Objects

This model has only one agent: grass.

#### Grass

The grass entity is defined as follows:

```yaml
grass:
  coordinates: (x, y)
  growth_stage: 0|1
  growth_countdown: float
  regrowth_time: float
  nutritional_value: float
```

The `coordinates` property depicts the current position of the predator in the
two-dimensional grid. It is initialized from a CSV file that contains a list of
all 1600 coordinates.

The `growth_stage` property stores the current growth stage of the grass: 0
means it is growing, and 1 means it is fully grown.

The `growth_countdown` property stores the number of steps after which the grass
becomes fully grown. The `regrowth_time` property is static and learnable, and
stores the max value of the countdown property.

The `nutritional_value` property is a static but learnable property that stores
the amount of energy gained by a predator when it consumes a single prey entity.

## Network

The model makes use of the adjacency matrix of a two-dimensional grid filled
with predator and prey to simulate the movement of those entities.

```yaml
network:
  agent_agent:
    grid: [predator, prey]
```

## Substeps

Each substep is a `torch.nn.ModuleDict` that takes an input state, and produces
an updated state as output. A substep consists of three phases:

1. Observation (retrieving relevant information from the state)
2. Policy/Action (deciding on the course of action as per the observations)
3. Transition (randomizing and updating the state according to the action)

This model consists of four substeps: `move`, `eat_grass`, `hunt_prey`, and
`grow_grass`.

<details>
  <summary>Helper functions</summary>

```python
# define all the helper functions we need.

def get_neighbors(pos, adj_grid, bounds):
  """
    Returns a list of neighbours for each position passed in the given
    `pos` tensor, using the adjacency matrix passed in `adj_grid`.
  """
  x, y = pos
  max_x, max_y = bounds

  # calculate the node number from the x, y coordinate.
  # each item (i, j) in the adjacency matrix, if 1 depicts
  # that i is connected to j and vice versa.
  node = (max_y * x) + y
  conn = adj_grid[node]

  neighbors = []
  for idx, cell in enumerate(conn):
    # if connected, calculate the (x, y) coords of the other
    # node and add it to the list of neighbors.
    if cell == 1:
      c = (int) (idx % max_y)
      r = math.floor((idx - c) / max_y)

      neighbors.append(
        [torch.tensor(r), torch.tensor(c)]
      )

  return torch.tensor(neighbors)

# define a function to retrieve the input required
def get_find_neighbors_input(state, input_variables):
    bounds = get_var(state, input_variables['bounds'])
    adj_grid = get_var(state, input_variables['adj_grid'])
    positions = get_var(state, input_variables['positions'])

    return bounds, adj_grid, positions

def get_decide_movement_input(state, input_variables):
    positions = get_var(state, input_variables['positions'])
    energy = get_var(state, input_variables['energy'])

    return positions, energy

def get_update_positions_input(state, input_variables):
    prey_energy = get_var(state, input_variables['prey_energy'])
    pred_energy = get_var(state, input_variables['pred_energy'])
    prey_work = get_var(state, input_variables['prey_work'])
    pred_work = get_var(state, input_variables['pred_work'])

    return prey_energy, pred_energy, prey_work, pred_work

def get_find_eatable_grass_input(state, input_variables):
    bounds = get_var(state, input_variables['bounds'])
    positions = get_var(state, input_variables['positions'])
    grass_growth = get_var(state, input_variables['grass_growth'])

    return bounds, positions, grass_growth

def get_eat_grass_input(state, input_variables):
    bounds = get_var(state, input_variables['bounds'])
    prey_pos = get_var(state, input_variables['prey_pos'])
    energy = get_var(state, input_variables['energy'])
    nutrition = get_var(state, input_variables['nutrition'])
    grass_growth = get_var(state, input_variables['grass_growth'])
    growth_countdown = get_var(state, input_variables['growth_countdown'])
    regrowth_time = get_var(state, input_variables['regrowth_time'])

    return bounds, prey_pos, energy, nutrition, grass_growth, growth_countdown, regrowth_time

def get_find_targets_input(state, input_variables):
    prey_pos = get_var(state, input_variables['prey_pos'])
    pred_pos = get_var(state, input_variables['pred_pos'])

    return prey_pos, pred_pos

def get_hunt_prey_input(state, input_variables):
    prey_pos = get_var(state, input_variables['prey_pos'])
    prey_energy = get_var(state, input_variables['prey_energy'])
    pred_pos = get_var(state, input_variables['pred_pos'])
    pred_energy = get_var(state, input_variables['pred_energy'])
    nutrition = get_var(state, input_variables['nutritional_value'])

    return prey_pos, prey_energy, pred_pos, pred_energy, nutrition

def get_grow_grass_input(state, input_variables):
    grass_growth = get_var(state, input_variables['grass_growth'])
    growth_countdown = get_var(state, input_variables['growth_countdown'])

    return grass_growth, growth_countdown
```

</details>

### Move

First, we **observe** the state, and find a list of neighboring positions for
each of the predators/prey currently alive.

```python
@Registry.register_substep("find_neighbors", "observation")
class FindNeighbors(SubstepObservation):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, state):
    bounds, adj_grid, positions = get_find_neighbors_input(state, self.input_variables)

    # for each agent (prey/predator), find the adjacent cells and pass
    # them on to the policy class.
    possible_neighbors = []
    for pos in positions:
      possible_neighbors.append(
        get_neighbors(pos, adj_grid, bounds)
      )

    return { self.output_variables[0]: possible_neighbors }
```

Then, we decide the course of **action**: to move each entity to a random
neighboring position, only if they have the energy to do so.

```python
@Registry.register_substep("decide_movement", "policy")
class DecideMovement(SubstepAction):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, state, observations):
    positions, energy = get_decide_movement_input(state, self.input_variables)
    possible_neighbors = observations['possible_neighbors']

    # randomly choose the next position of the agent. if the agent
    # has non-positive energy, don't let it move.
    next_positions = []
    for idx, pos in enumerate(positions):
      next_positions.append(
        random.choice(possible_neighbors[idx]) if energy[idx] > 0 else pos
      )

    return { self.output_variables[0]: torch.stack(next_positions, dim=0) }
```

Lastly, we **update** the state, with the new positions of the entities, and
reduce the energy of each entity by the value of the `stride_work` learnable
parameter.

```python
@Registry.register_substep("update_positions", "transition")
class UpdatePositions(SubstepTransition):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, state, action):
    prey_energy, pred_energy, prey_work, pred_work = get_update_positions_input(state, self.input_variables)

    # reduce the energy of the agent by the work required by them
    # to take one step.
    prey_energy = prey_energy + torch.full(prey_energy.shape, -1 * (prey_work.item()))
    pred_energy = pred_energy + torch.full(pred_energy.shape, -1 * (pred_work.item()))

    return {
      self.output_variables[0]: action['prey']['next_positions'],
      self.output_variables[1]: prey_energy,
      self.output_variables[2]: action['predator']['next_positions'],
      self.output_variables[3]: pred_energy
    }
```

### Eat

First, **decide** which grass is fit to be consumed by the prey.

```python
@Registry.register_substep("find_eatable_grass", "policy")
class FindEatableGrass(SubstepAction):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, state, observations):
    bounds, positions, grass_growth = get_find_eatable_grass_input(state, self.input_variables)

    # if the grass is fully grown, i.e., its growth_stage is equal to
    # 1, then it can be consumed by prey.
    eatable_grass_positions = []
    max_x, max_y = bounds
    for pos in positions:
      x, y = pos
      node = (max_y * x) + y
      if grass_growth[node] == 1:
        eatable_grass_positions.append(pos)

    # pass on the consumable grass positions to the transition class.
    return { self.output_variables[0]: eatable_grass_positions }
```

Then, simulate the consumption of the grass, and **update** the growth stage,
growth countdown, and energies of the grass and prey respectively.

```python
@Registry.register_substep("eat_grass", "transition")
class EatGrass(SubstepTransition):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, state, action):
    bounds, prey_pos, energy, nutrition, grass_growth, growth_countdown, regrowth_time = get_eat_grass_input(state, self.input_variables)

    # if no grass can be eaten, skip modifying the state.
    if len(action['prey']['eatable_grass_positions']) < 1:
      return {}

    eatable_grass_positions = torch.stack(action['prey']['eatable_grass_positions'], dim=0)
    max_x, max_y = bounds
    energy_mask = None
    grass_mask, countdown_mask = torch.zeros(*grass_growth.shape), torch.zeros(*growth_countdown.shape)

    # for each consumable grass, figure out if any prey agent is at
    # that position. if yes, then mark that position in the mask as
    # true. also, for all the grass that will be consumed, reset the
    # growth stage.
    for pos in eatable_grass_positions:
      x, y = pos
      node = (max_y * x) + y

      # TODO: make sure dead prey cannot eat
      e_m = (pos == prey_pos).all(dim=1).view(-1, 1)
      if energy_mask is None:
        energy_mask = e_m
      else:
        energy_mask = e_m + energy_mask

      grass_mask[node] = -1
      countdown_mask[node] = regrowth_time - growth_countdown[node]

    # energy + nutrition adds the `nutrition` tensor to all elements in
    # the energy tensor. the (~energy_mask) ensures that the change is
    # undone for those prey that did not consume grass.
    energy = energy_mask*(energy + nutrition) + (~energy_mask)*energy

    # these masks use simple addition to make changes to the original
    # values of the tensors.
    grass_growth = grass_growth + grass_mask
    growth_countdown = growth_countdown + countdown_mask

    return {
      self.output_variables[0]: energy,
      self.output_variables[1]: grass_growth,
      self.output_variables[2]: growth_countdown
    }
```

### Hunt

First, **decide** which prey are to be eaten.

```python
@Registry.register_substep("find_targets", "policy")
class FindTargets(SubstepAction):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, state, observations):
    prey_pos, pred_pos = get_find_targets_input(state, self.input_variables)

    # if there are any prey at the same position as a predator,
    # add them to the list of targets to kill.
    target_positions = []
    for pos in pred_pos:
      if (pos == prey_pos).all(-1).any(-1) == True:
        target_positions.append(pos)

    # pass that list of targets to the transition class.
    return { self.output_variables[0]: target_positions }
```

Then, **update** the energies of both the prey and the predator.

```python
@Registry.register_substep("hunt_prey", "transition")
class HuntPrey(SubstepTransition):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, state, action):
    prey_pos, prey_energy, pred_pos, pred_energy, nutrition = get_hunt_prey_input(state, self.input_variables)

    # if there are no targets, skip the state modifications.
    if len(action['predator']['target_positions']) < 1:
      return {}

    target_positions = torch.stack(action['predator']['target_positions'], dim=0)

    # these are masks similars to the ones in `substeps/eat.py`.
    prey_energy_mask = None
    pred_energy_mask = None
    for pos in target_positions:
      pye_m = (pos == prey_pos).all(dim=1).view(-1, 1)
      if prey_energy_mask is None:
        prey_energy_mask = pye_m
      else:
        prey_energy_mask = prey_energy_mask + pye_m

      pde_m = (pos == pred_pos).all(dim=1).view(-1, 1)
      if pred_energy_mask is None:
        pred_energy_mask = pde_m
      else:
        pred_energy_mask = pred_energy_mask + pde_m

    # any prey that is marked for death should be given zero energy.
    prey_energy = prey_energy_mask*0 + (~prey_energy_mask)*prey_energy
    # any predator that has hunted should be given additional energy.
    pred_energy = pred_energy_mask*(pred_energy + nutrition) + (~pred_energy_mask)*pred_energy

    return {
      self.output_variables[0]: prey_energy,
      self.output_variables[1]: pred_energy
    }
```

### Grow

In this substep, we simply **update** the growth countdown of every grass
object, and if the countdown has elapsed, we update the growth stage to `1`.

```python
@Registry.register_substep("grow_grass", "transition")
class GrowGrass(SubstepTransition):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, state, action):
    grass_growth, growth_countdown = get_grow_grass_input(state, self.input_variables)

    # reduce all countdowns by 1 unit of time.
    growth_countdown_mask = torch.full(growth_countdown.shape, -1)
    growth_countdown = growth_countdown + growth_countdown_mask

    # if the countdown has reached zero, set the growth stage to 1,
    # otherwise, keep it zero.
    grass_growth_mask = (growth_countdown <= 0).all(dim=1)
    grass_growth = grass_growth_mask*(1) + (~grass_growth_mask)*(0)

    return {
      self.output_variables[0]: grass_growth.view(-1, 1),
      self.output_variables[1]: growth_countdown
    }
```

## Execution: Configuration, Registry, and Runner

### Configuration

There are several parts to the configuration, written in a file traditionally
called `config.yaml`. The following is a brief overview of all the major
sections in the configuration file.

```yaml
# config.yaml
# configuration for the predator-prey model.

metadata:
  # device type, episode count, data files, etc.

state:
  environment:
    # variables/properties of the simulated enviroment.

  agents:
    # a list of agents in the simulation, and their properties.
    # each property must be initialized by specifying a value
    # or a generator function, and have a fixed tensor shape.

  objects:
    # a list of objects, similar to the agents list.

  network:
    # a list of interaction models for the simulation.
    # could be a grid, or a directed graph, etc.

substeps:
  # a list of substeps
  # each substep has a list of agents to run that substep for
  # as well as the function, input and output variables for each
  # part of that substep (observation, policy and transition)
```

The following is an example of defining a property in the configuration.

```yaml
bounds:
  name: 'Bounds'
  learnable: false
  shape: 2
  dtype: 'int'
  value:
    - ${simulation_metadata.max_x} # you can refer to other parts of the config using
    - ${simulation_metadata.max_y} # the template syntax, i.e., ${path.to.config.value}
  initialization_function: null
```

Notice that to define one single property, we mentioned:

- the name of the property, here, `'bounds'`.
- whether or not the property is learnable, in this case, `false`.
- the shape of the tensor that stores the values, in this case, it is a
  one-dimensional array of two elements: `(max_x, max_y)`.
- the value of the property, either by directly providing the value or by
  providing a function that returns the value.

The full configuration for the predator-prey model can be found
[here](../config.yaml).

```python
# define helper functions used in the configuration

@Registry.register_helper('map', 'network')
def map_network(params):
  coordinates = (40.78264403323726, -73.96559413265355) # central park
  distance = 550

  graph = ox.graph_from_point(coordinates, dist=distance, simplify=True, network_type="walk")
  adjacency_matrix = nx.adjacency_matrix(graph).todense()

  return graph, torch.tensor(adjacency_matrix)

@Registry.register_helper('random_float', 'initialization')
def random_float(shape, params):
  """
    Generates a `Tensor` of the given shape, with random floating point
    numbers in between and including the lower and upper limit.
  """

  max = params['upper_limit'] + 1 # include max itself.
  min = params['lower_limit']

  # torch.rand returns a tensor of the given shape, filled with
  # floating point numbers in the range (0, 1]. multiplying the
  # tensor by max - min and adding the min value ensure it's
  # within the given range.
  tens = (max - min) * torch.rand(shape) + min

  return tens

@Registry.register_helper('random_int', 'initialization')
def random_int(shape, params):
  """
    Generates a `Tensor` of the given shape, with random integers in
    between and including the lower and upper limit.
  """

  max = math.floor(params['upper_limit'] + 1) # include max itself.
  min = math.floor(params['lower_limit'])

  # torch.randint returns the tensor we need.
  tens = torch.randint(min, max, shape)

  return tens
```

### Registry and Runner

The code that **executes** the simulation uses the AgentTorch `Registry` and
`Runner`, like so:

```python
config = read_config('config-map.yaml')
metadata = config.get('simulation_metadata')
num_episodes = metadata.get('num_episodes')
num_steps_per_episode = metadata.get('num_steps_per_episode')
num_substeps_per_step = metadata.get('num_substeps_per_step')
```

The registry is stores all the classes and functions used by the model, and
allows the runner to call them as needed when intializing the simulation and
executing the substeps.

```python
registry = Registry()
registry.register(read_from_file, 'read_from_file', 'initialization')
registry.register(grid_network, 'grid', key='network')
```

The runner intializes and executes the simulation for us. It also returns:

- a list of the learnable parameters, so we can run optimization functions on
  them and use the optimized values for the next episode.
- the trajectory of the state so far, so we can visualize the state using
  libraries like `matplotlib`.

```python
runner = Runner(config, registry)
```

<small> The source code for the visualizer used in the following block is given
in the next section. </small>

```python
runner.init()

for episode in range(num_episodes):
  runner.step(num_steps_per_episode)

  final_states = list(filter(
    lambda x: x['current_substep'] == str(num_substeps_per_step - 1),
    runner.state_trajectory[-1]
  ))
  visualizer = Plot(metadata.get('max_x'), metadata.get('max_y'))
  visualizer.plot(final_states)
```

![png](../../media/predator-prey_47_0.png)

![png](../../media/predator-prey_47_1.png)

![png](../../media/predator-prey_47_2.png)

![png](../../media/predator-prey_47_3.png)

![png](../../media/predator-prey_47_4.png)

![png](../../media/predator-prey_47_5.png)

## Visualization

You can plot the simulation in different ways. In this notebook, two such
methods are demonstrated; the X-Y grid, and the OpenStreetMap plot.

```python
# display the gifs

from IPython.display import HTML

HTML("""
    <table>
    <tr><td>
    <video alt="grid" autoplay>
        <source src="../predator-prey.mp4" type="video/mp4">
    </video>
    </td><td>
    <img src="../predator-prey.gif" alt="map" />
    </td></tr>
    </table>
""")
```

<table>
<tr><td>
<video alt="grid" autoplay>
    <source src="../predator-prey.mp4" type="video/mp4">
</video>
</td><td>
<img src="../predator-prey.gif" alt="map" />
</td></tr>
</table>

```python
# render the map

from IPython.display import display, clear_output

import time
import matplotlib
import matplotlib.pyplot as plotter
import matplotlib.patches as patcher
import contextily as ctx

%matplotlib inline

class Plot:
  def __init__(self, max_x, max_y):
    # intialize the scatterplot
    self.figure, self.axes = None, None
    self.prey_scatter, self.pred_scatter = None, None
    self.max_x, self.max_y = max_x, max_y

    plotter.xlim(0, max_x - 1)
    plotter.ylim(0, max_y - 1)
    self.i = 0

  def update(self, state):
    graph = state['network']['agent_agent']['predator_prey']['graph']
    self.coords = [(node[1]['x'], node[1]['y']) for node in graph.nodes(data=True)]
    self.coords.sort(key=lambda x: -(x[0] + x[1]))

    self.figure, self.axes = ox.plot_graph(graph, edge_linewidth=0.3, edge_color='gray', show=False, close=False)
    ctx.add_basemap(self.axes, crs=graph.graph['crs'], source=ctx.providers.OpenStreetMap.Mapnik)
    self.axes.set_axis_off()

    # get coordinates of all the entities to show.
    prey = state['agents']['prey']
    pred = state['agents']['predator']
    grass = state['objects']['grass']

    # agar energy > 0 hai... toh zinda ho tum!
    alive_prey = prey['coordinates'][torch.where(prey['energy'] > 0)[0]]
    alive_pred = pred['coordinates'][torch.where(pred['energy'] > 0)[0]]
    # show only fully grown grass, which can be eaten.
    grown_grass = grass['coordinates'][torch.where(grass['growth_stage'] == 1)[0]]

    alive_prey_x, alive_prey_y = np.array([
      self.coords[(self.max_y * pos[0]) + pos[1]] for pos in alive_prey
    ]).T
    alive_pred_x, alive_pred_y = np.array([
      self.coords[(self.max_y * pos[0]) + pos[1]] for pos in alive_pred
    ]).T

    # show prey in dark blue and predators in maroon.
    self.axes.scatter(alive_prey_x, alive_prey_y, c='#0d52bd', marker='.')
    self.axes.scatter(alive_pred_x, alive_pred_y, c='#8b0000', marker='.')

    # increment the step count.
    self.i += 1
    # show the current step count, and the population counts.
    self.axes.set_title('Predator-Prey Simulation #' + str(self.i), loc='left')
    self.axes.legend(handles=[
      patcher.Patch(color='#fc46aa', label=str(self.i) + ' step'),
      patcher.Patch(color='#0d52bd', label=str(len(alive_prey)) + ' prey'),
      patcher.Patch(color='#8b0000', label=str(len(alive_pred)) + ' predators'),
      # patcher.Patch(color='#d1ffbd', label=str(len(grown_grass)) + ' grass')
    ])

    display(plotter.gcf())
    clear_output(wait=True)
    time.sleep(1)

  def plot(self, states):
    # plot each state, one-by-one
    for state in states:
      self.update(state)

    clear_output(wait=True)
```

---

## Document: tutorials/integrating-with-beckn/index.md

# AgentTorch-Beckn Solar Model

## Overview

AgentTorch is a differentiable learning framework that enables you to run simulations with
over millions of autonomous agents. [Beckn](https://becknprotocol.io) is a protocol that
enables the creation of open, peer-to-peer decentralized networks for pan-sector economic
transactions.

This model integrates Beckn with AgentTorch, to simulate a solar energy network in which
households in a locality can decide to either buy solar panels and act as providers of
solar energy, or decide to use the energy provided by other households instead of
installing solar panels themselves.

> ![A visualization of increase in net solar energy used per street](./visualization.gif)
>
> A visualization of increase in net solar energy used per street.

## Mapping Beckn Protocol to AgentTorch

### 1. Network

The participants in the Beckn network (providers, customers and gateways) are considered
agents that interact with each other.

### 2. Operations

The following operations are simulated as substeps:

##### 1. a customer will `search` and `select` a provider

- the customer selects the closest provider with the least price

##### 2. the customer will `order` from the provider

- the customer orders basis their monthly energy demand
- the provider only confirms the order if it has the capacity to

##### 3. the provider will `fulfill` the order

- the provider's capacity is reduced for the given step (~= 30 real days)

##### 4. the customer will `pay` for the work done

- the provider's revenue is incremented, while the customer's wallet is deducted the same
  amount.
- the amount to be paid is determined by the provider's price, multiplied by the amount of
  energy supplied.

##### 5. the provider will `restock` their solar energy

- the amount of energy replenished SHOULD BE (TODO) dependent on the season as well as the
  weather.

Each of the substeps' code (apart from #5) is taken as-is from the
[AgentTorch Beckn integration](https://github.com/AgentTorch/agent-torch-beckn).

> Note that while Beckn's API calls are asynchronous, the simulation assumes they are
> synchronous for simplicity.

### 3. Data

The data for this example model is currently sourced from various websites, mostly from
[data.boston.gov](http://data.boston.gov/). However, the data should actually come from
the Beckn Protocol's implementation of a solar network.

## Running the Model

To run the model, clone the github repository first:

```python
# git clone --depth 1 --branch solar https://github.com/AgentTorch/agent-torch-beckn solar-netowkr
```

Then, setup a virtual environment and install all dependencies:

```python
# cd solar-network/
# python -m venv .venv/bin/activate
# . .venv/bin/activate
# pip install -r requirements.txt
```

Once that is done, you can edit the configuration ([`config.yaml`](../config.yaml)), and
change the data used in the simulation by editing the simulation's data files
([`data/simulator/{agent}/{property}.csv`](../data/simulator/)).

Then, open Jupyter Lab and open the `main.ipynb` notebook, and run all the cells.

```python
# pip install jupyterlab
# jupyter lab
```

## Todos

- Add more visualizations (plots/graphs/heatmaps/etc.)
- Improve the data used for the simulation, reduce the number of random values.
- Add more detailed logic to the substeps, i.e., seasonal fluctuation in energy generation
  and prices.
- Include and run a sample beckn instance to pull fake data from.

---

## Document: tutorials/creating-archetypes/index.md

## Archetype Tutorial

#### Step 1: Setup
First, let's set up our environment and import the necessary libraries:


```python
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.behavior import Behavior
from agent_torch.populations import NYC
from agent_torch.core.llm.backend import LangchainLLM
OPENAI_API_KEY = None
```

Setup : Covid Cases Data and Unemployment Rate


```python
from utils import get_covid_cases_data
csv_path = '/models/covid/data/county_data.csv'
monthly_cases_kings = get_covid_cases_data(csv_path=csv_path,county_name='Kings County')

```

#### Step 2: Initialise LLM Instance

We can use either of the Langchain and Dspy backends to initialise a LLM instance. While these are the frameworks we are supporting currently, you may choose to use your own framework of choice by extending the LLMBackend class provided with AgentTorch.

Let's see how we can use Langchain to initialise an LLM instance

GPT 3.5 Turbo


```python
agent_profile = "You are an helpful agent who is trying to help the user make a decision. Give answer as a single number between 0 and 1, only."
llm_langchain_35 = LangchainLLM(
    openai_api_key=OPENAI_API_KEY, agent_profile=agent_profile, model="gpt-3.5-turbo"
)
```

#### Step 3: Define an Archetype


```python
# Create an object of the Archetype class
# n_arch is the number of archetypes to be created. This is used to calculate a distribution from which the outputs are then sampled.
archetype_n_2 = Archetype(n_arch=2) 
archetype_n_12 = Archetype(n_arch=12)
```

Create an object of the Behavior class


```python
# Define a prompt template
# Age,Gender and other attributes which are part of the population data, will be replaced by the actual values of specified region, during the simulation.
# Other variables like Unemployment Rate and COVID cases should be passed as kwargs to the behavior model.
user_prompt_template = "Your age is {age}, gender is {gender}, ethnicity is {ethnicity}, and the number of COVID cases is {covid_cases}.Current month is {month} and year is {year}."

# Create a behavior model
# You have options to pass any of the above created llm objects to the behavior class
# Specify the region for which the behavior is to be sampled. This should be the name of any of the regions available in the populations folder.
earning_behavior_n_2 = Behavior(
    archetype=archetype_n_2.llm(llm=llm_langchain_35, user_prompt=user_prompt_template),
    region=NYC
)
earning_behavior_n_12 = Behavior(
    archetype=archetype_n_12.llm(llm=llm_langchain_35, user_prompt=user_prompt_template),
    region=NYC
)
```


```python
# Define arguments to be used for creating a query for the LLM Instance
kwargs = {
    "month": "January",
    "year": "2020",
    "covid_cases": 1200,
    "device": "cpu",
    "current_memory_dir": "/populations/astoria/conversation_history",
    "unemployment_rate": 0.05,
}
```

#### Step 4: Compare performance between different Configurations of Archetype


```python
from utils import get_labor_data, get_labor_force_correlation

labor_force_df_n_2, observed_labor_force_n_2, correlation_n_2 = get_labor_force_correlation(
    monthly_cases_kings, 
    earning_behavior_n_2, 
    'agent_torch/models/macro_economics/data/unemployment_rate_csvs/Brooklyn-Table.csv',
    kwargs
)
labor_force_df_n_12, observed_labor_force_n_12, correlation_n_12 = get_labor_force_correlation(
    monthly_cases_kings, 
    earning_behavior_n_12, 
    'agent_torch/models/macro_economics/data/unemployment_rate_csvs/Brooklyn-Table.csv',
    kwargs
)
print(f"Correlation with 2 Archetypes is {correlation_n_2} and 12 Archetypes is {correlation_n_12}")
```

---

## Document: tutorials/differentiable-discrete-sampling/index.md

# Differentiable Discrete Sampling using AgentTorch     
## Introduction
Discrete sampling poses significant challenges in gradient-based optimization due to its non-differentiable nature, which prevents effective backpropagation. Operations like argmax disrupt gradient flow, leading to high-variance or biased gradient estimates. The Gumbel-Softmax technique addresses this by using a reparameterization trick that adds Gumbel noise to logits and applies a temperature-controlled softmax, enabling differentiable approximations of discrete samples. As the temperature approaches zero, the method produces near-discrete outputs while maintaining gradient flow, making it suitable for integrating discrete sampling into neural networks.

## Rethinking Discrete Sampling
It was assumed that Gumbel softmax solves this problem. However, Gumbel-Softmax has its own limitations. The temperature parameter introduces a bias-variance tradeoff: higher temperatures smooth gradients but deviate from true categorical distributions, while lower temperatures yield near-discrete samples with unstable gradients. Additionally, its continuous approximations may require straight-through estimators, which can introduce bias during backpropagation. These issues make Gumbel-Softmax less effective in tasks requiring precise distribution matching or structured outputs, highlighting the need for further improvements in discrete sampling techniques.

So we introduce a new method for discrete sampling using the `agent_torch.core.distribution.Categorical` class. This class provides a differentiable approximation to discrete sampling, allowing for gradient-based optimization while maintaining the integrity of the categorical distribution.

This estimator can simply be used as follows:

```python
import torch
from agent_torch.core.distributions import Categorical
# Define the probabilities for each category
probs = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float32)
# Create a Categorical distribution
sample = Categorical.apply(probs)
# The sample will be a tensor containing the sampled category
print(sample)
```
Let's discuss more about this by seeing its application in various experiments.
## Experiment 1: Random Walk
Let's implement a 1D markovian random walk X0, X1, ...., Xn using the `agent_torch.core.distribution.Categorical` sampling method. The agent can move left or right with probabilites:

- Xn+1 = Xn + 1 with probability e^(-Xn/p)
- Xn+1 = Xn - 1 with probability 1 - e^(-Xn/p)

First, lets import the important modules:

```python
import torch
import math
from agent_torch.core.distributions import Categorical
```
We are interested in studying the asymptotic behavior of the variance of
our automatically derived gradient estimator, and so set p = n so that the transition function varies appreciably over the range of the walk for all n.

Let's define the main function:

```python
def random_walk_categorical(n, p, device):
    x = 0.0  # initial state
    path = [0.0]
    for _ in range(n):
        # Compute the probability of moving up.
        q = math.exp(-x / p)
        prob = torch.tensor([q, 1.0 - q], dtype=torch.float32, device=device).unsqueeze(0)  
        # Sample an action using the custom Categorical function.
        sample = Categorical.apply(prob)  
        move = 1 if sample.item() == 0 else -1
        # if at x==0, a downward move is overridden, since probability for going up is 1.
        if x == 0 and move == -1:
            move = 1
        x += move
        path.append(x)
    return path
```
This random walk can be generated by:

```python
n = 20  # A 20 step simulation
p = n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_path = random_walk_categorical(n,p,device)

# This random walk looks like [0,1,2,1,...]
```
Having seen how a random walk is implemented using AgentTorch, let's benchmark this against the Gumbel softmax method. The Gumbel softmax method is a differentiable approximation to the categorical distribution, allowing for gradient-based optimization. Let's discuss the experiment setup.

#### Experiment Setup
This experiment focuses on the optimization of a parameter θ (theta) embedded within an exponential probability distribution function exp(-(x + θ)/p), which governs the stochastic transition dynamics of our model. The primary objective is to calibrate θ such that the model's behavior closely approximates a baseline implementation, as measured by mean squared error (MSE).

The methodology involves generating a substantial dataset comprising 1,000 trajectories, each consisting of 100 discrete time steps. This dataset is partitioned following standard machine learning protocols, with 70% allocated for parameter estimation (training) and 30% reserved for out-of-sample validation (testing).

By systematically adjusting θ, we aim to modulate the underlying probability distribution, thereby altering the likelihood of specific state transitions. This parameter optimization process seeks to minimize the discrepancy between the simulated trajectories and those produced by the baseline model. The efficacy of each candidate value for θ is quantitatively assessed via the MSE metric, which provides a rigorous measure of the deviation between the predicted and reference trajectories.

This approach enables the fine-tuning of stochastic models to replicate observed phenomena with enhanced precision, with potential applications in various domains including statistical physics, financial modeling, and computational biology.

#### Results
The empirical findings demonstrate that the `agent_torch.core.distribution.Categorical` approach consistently exhibits superior performance metrics compared to the Gumbel-based method. Specifically, the `agent_torch.core.distribution.Categorical` method maintains consistently lower Wasserstein distance values across all experimental configurations, indicating better alignment between simulated and baseline distributions. Furthermore, the `agent_torch.core.distribution.Categorical` approach effectively preserves the variance ratio at approximately unity, which substantiates that the generated trajectories maintain distributional characteristics highly comparable to those of the baseline.

Although the parameter convergence behavior varies across different initialization points, particularly for initial values of 10.0 and 0.0, the distributional properties of the Categorical method's outputs remain demonstrably superior to those produced by the Gumbel approach. This superiority is quantitatively verified through both lower Wasserstein distance measurements and reduced mean squared error metrics, which collectively indicate that the `agent_torch.core.distribution.Categorical` method generates distributions with greater fidelity to the baseline distribution regardless of initialization conditions. These results suggest that the `agent_torch.core.distribution.Categorical` approach provides a more robust framework for distribution matching in this experimental context, maintaining consistent performance advantages across varied experimental configurations.

These results will further become clear when we plot these random walks. It can clearly be infered that the Gumbel method starts diverging from the baseline and performs poorly on the test dataset.

![image](rwalk.png)
![image](rwalk1.png)

First, we run the experiment for 100 time-steps and calibrate theta values for both methods. Among the methods, `agent_torch.core.distribution.Categorical` stays relatively close to the baseline, while the Gumbel-based approach begins to drift early and deviates substantially in the testing region. This suggests that the Categorical method generalizes better across regions and is more stable under extended evaluation.

Second, we extend the experiment to 1000 steps to examine long-term behavior. Over this longer horizon, the difference becomes even more pronounced. Gumbel's trajectory continues to diverge and accumulates a large positional error, confirming its poor generalization performance. In contrast, `agent_torch.core.distribution.Categorical` remains much more aligned with the baseline throughout.

## Experiment 2: Neural Relational Inference
The neural relational inference experiment is designed to infer and model latent interactions among entities in a dynamic system. In this experiment, a graph-based neural architecture is employed in which a factor graph CNN encoder extracts relational features from observed data, while a learned recurrent interaction net decoder predicts future states by modeling interactions between nodes (or atoms). The goal is to simultaneously learn the underlying relations and use these learned interactions to improve prediction accuracy and interpretability of the system’s dynamics.

#### Experiment Setup

The NRI experiment specifically focuses on learning to infer interactions in physical systems without supervision. The model is structured as a variational auto-encoder where the latent code represents the underlying interaction graph and the reconstruction is based on graph neural networks. The researchers conducted experiments on simulated physical systems including springs and charged particles. The model is evaluated on its ability to recover ground-truth interactions in these simulated environments, as well as its capacity to find interpretable structure and predict complex dynamics in real-world data such as motion capture and sports tracking data

Initially, the experiment employed a Gumbel-Softmax approach for discrete sampling. In this setup, the addition of Gumbel noise and a temperature-controlled softmax allowed for differentiable approximations of categorical samples. However, the inherent bias-variance tradeoff—where higher temperatures yield smoother but less discrete gradients, and lower temperatures produce near-discrete but unstable gradients—limits the method's effectiveness. While the negative log-likelihood decreases over epochs, the KL divergence remains relatively low, suggesting insufficient regularization of the discrete structure .

Recognizing these limitations, the experiment was repeated using our `agent_torch.core.distribution.Categorical` class. This new estimator directly provides a differentiable approximation for discrete sampling, bypassing some of the drawbacks inherent in the Gumbel-Softmax method. Notably, by more tightly coupling the sampling process to the categorical distribution, the estimator mitigates the bias-variance issue and improves gradient stability during training.

#### Results
![acc](acc_nri.png)
![kl](kl_nri.png)

The training logs for the categorical estimator experiment reveal several improvements:

- Stable KL Divergence: The KL divergence values remained consistent from early epochs into convergence. This higher and stable KL value suggests that the model is enforcing a stronger regularization on the inferred discrete relations, leading to a more consistent latent structure.
- Lower Negative Log-Likelihood: While both methods converge to low nll_train values as training proceeds, the categorical estimator maintains comparably low loss values alongside improved training accuracy. 
- Improved Predictive Accuracy: The accuracy trends in the logs show that the categorical estimator experiment reaches and sustains higher accuracy levels. The results point to a model that not only fits the data better but also generalizes more effectively—an essential trait when dealing with structured, relational data.

## Conclusion
This tutorial demonstrated how to implement and use differentiable discrete sampling operations using AgentTorch.
---

