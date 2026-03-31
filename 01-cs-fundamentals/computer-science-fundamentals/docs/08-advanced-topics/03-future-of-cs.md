# The Future of Computer Science

> "The best way to predict the future is to invent it." — Alan Kay

## What You Will Learn in This Chapter

- [ ] Understand the fundamental principles of quantum computing
- [ ] Learn the frontiers of AI/ML and future directions
- [ ] Grasp new computing paradigms
- [ ] Understand next-generation security including post-quantum cryptography and zero trust
- [ ] Envision transformations in programming and new development styles
- [ ] Learn about new domains such as brain-computer interfaces and space computing


## Prerequisites

Having the following knowledge before reading this guide will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Blockchain Basics](./02-blockchain-basics.md)

---

## 1. Quantum Computing

```
Classical Computers vs Quantum Computers:

  Classical bit: 0 or 1 (deterministic)

  Quantum bit (qubit): Superposition of 0 and 1
  |psi> = alpha|0> + beta|1>
  -> Both 0 and 1 until observed

  n qubits = represent 2^n states simultaneously
  +--------------------------------------+
  | Classical 10 bits:  1 of 1,024       |
  | Quantum 10 qubits:  hold all 1,024  |
  |                                      |
  | Classical 300 bits:  1 of 2^300      |
  | Quantum 300 qubits:  hold all 2^300 |
  | (2^300 > number of atoms in universe)|
  +--------------------------------------+

Three Fundamental Principles of Quantum:

  1. Superposition:
     A single qubit represents 0 and 1 simultaneously
     -> Explores many computational paths in parallel

  2. Entanglement:
     Two qubits become correlated
     -> Observing one instantly determines the other's state
     -> "Spooky action at a distance" (Einstein)

  3. Interference:
     Amplifies the probability of correct answers and cancels out wrong ones
     -> The core of quantum algorithms

Quantum Gates:
  Classical: AND, OR, NOT
  Quantum: Hadamard(H), CNOT, Toffoli, Phase
  -> Combined in quantum circuits to form algorithms
```

### 1.1 Quantum Algorithms

```
Quantum Algorithms:

  Shor's Algorithm (1994, Peter Shor):
  - Factors large integers efficiently
  - Classical: exponential time -> Quantum: polynomial time
  - Threatens the security of RSA encryption
  - Currently, sufficient qubit counts have not been achieved

  Grover's Algorithm (1996, Lov Grover):
  - Search in unsorted data
  - Classical: O(N) -> Quantum: O(sqrt(N))
  - Quadratic speedup (not exponential)

  Quantum Machine Learning:
  - Quantum kernel methods
  - Variational Quantum Eigensolver (VQE)
  - Quantum Approximate Optimization Algorithm (QAOA)

  Quantum Simulation:
  - Directly simulate quantum mechanical behavior of molecules and materials
  - Exponentially difficult for classical computers
  - Potential to revolutionize drug discovery and materials design
  - Example: Design of nitrogen fixation catalysts (fertilizer production efficiency)

  Quantum Key Distribution (QKD):
  - BB84 protocol: Key distribution based on quantum mechanical principles
  - Eavesdropping disturbs quantum states and is therefore detectable
  - "Security based on the laws of physics"
```

### 1.2 Current State and Challenges of Quantum Computing

```
Current State (as of 2025):
  +----------------------------------------+
  | IBM:         1,000+ qubits (Eagle/Condor)|
  | Google:      Demonstrated quantum        |
  |              supremacy (Sycamore)         |
  | Challenges:  Noise, error rates,         |
  |              decoherence                  |
  | NISQ era:    Noisy intermediate-scale     |
  |              quantum devices              |
  | Practical    Cryptography breaking is     |
  | estimate:    10-20+ years away            |
  | Post-quantum NIST standardization in      |
  | crypto:      progress (CRYSTALS, etc.)    |
  +----------------------------------------+

Major Hardware Approaches:

  1. Superconducting (IBM, Google):
     - Operates at ultra-low temperatures (15mK ~ -273.135C)
     - Currently achieves the most qubits
     - Challenges: Large cooling equipment, short decoherence times

  2. Trapped Ion (IonQ, Honeywell/Quantinuum):
     - Traps individual ions with electromagnetic fields
     - High fidelity (low error rates)
     - Challenges: Slow gate speeds, difficult to scale

  3. Photonic (PsiQuantum, Xanadu):
     - Uses photons
     - Can operate at room temperature
     - Challenges: Photon loss, non-deterministic operations

  4. Topological (Microsoft):
     - Uses Majorana quasi-particles
     - Inherently high error resilience (theoretically)
     - Challenges: Reliable generation of Majorana quasi-particles still in demonstration stage

Importance of Error Correction:
  +------------------------------------------------+
  | Logical qubit = many physical qubits forming    |
  |                 one "reliable" qubit             |
  |                                                 |
  | Example: 1 logical qubit ~ 1,000-10,000         |
  |          physical qubits                         |
  | -> Practical quantum computation requires        |
  |    millions of physical qubits                   |
  | -> Currently: 1,000+ qubits -> still far from    |
  |    practical use                                 |
  |                                                 |
  | Surface Code:                                   |
  | - Error correction on a 2D lattice              |
  | - If physical qubit error rate is below          |
  |   threshold, logical error rate can be made      |
  |   arbitrarily small                              |
  +------------------------------------------------+

Quantum Computing Roadmap (IBM Quantum):
  2023: 1,000+ qubits (Condor)
  2025: Demonstration of error correction
  2029: Quantum-centric supercomputing
  2033+: Universal quantum computer
```

### 1.3 Practical Quantum Programming

```python
# Fundamentals of quantum programming with Qiskit

# Installation
# pip install qiskit qiskit-aer

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler

# === Example 1: Creating a Bell State (Quantum Entanglement) ===
def create_bell_state():
    """Create the most basic quantum entangled state"""
    qc = QuantumCircuit(2, 2)
    qc.h(0)        # Hadamard gate: |0> -> (|0> + |1>)/sqrt(2)
    qc.cx(0, 1)    # CNOT gate: creates entanglement
    qc.measure([0, 1], [0, 1])
    return qc

bell = create_bell_state()
print(bell.draw())
# Result: |00> and |11> each at 50%
# -> The two qubits are perfectly correlated

# === Example 2: Deutsch's Algorithm ===
def deutsch_algorithm(oracle_type='balanced'):
    """
    Determine if a function is constant or balanced with a single query
    Classical: requires 2 queries -> Quantum: 1 is sufficient
    """
    qc = QuantumCircuit(2, 1)

    # Prepare initial state
    qc.x(1)       # Set to |1>
    qc.h(0)       # Superposition
    qc.h(1)       # Superposition

    # Oracle (black-box function)
    if oracle_type == 'balanced':
        qc.cx(0, 1)  # f(x) = x (balanced function)
    # else: do nothing (f(x) = 0, constant function)

    # Hadamard before measurement
    qc.h(0)
    qc.measure(0, 0)

    return qc

# === Example 3: Quantum Teleportation ===
def quantum_teleportation():
    """
    Transfer a quantum state to another qubit
    Uses 2 bits of classical communication and pre-shared entanglement
    """
    qc = QuantumCircuit(3, 3)

    # Prepare the state to send (qubit 0)
    qc.rx(1.2, 0)  # Create an arbitrary state

    # Create a Bell pair (share qubits 1, 2)
    qc.h(1)
    qc.cx(1, 2)

    # Sender's operations
    qc.cx(0, 1)
    qc.h(0)

    # Measurement
    qc.measure([0, 1], [0, 1])

    # Receiver's correction operations (based on classical communication)
    qc.cx(1, 2)
    qc.cz(0, 2)

    qc.measure(2, 2)
    return qc

# === Example 4: Grover's Algorithm (2-qubit version) ===
def grover_2qubit(target='11'):
    """
    Search 4 elements in 1 query
    Classical: average 2 queries -> Quantum: 1
    """
    qc = QuantumCircuit(2, 2)

    # Initial superposition
    qc.h([0, 1])

    # Oracle: flip phase of target state
    if target == '11':
        qc.cz(0, 1)
    elif target == '00':
        qc.x([0, 1])
        qc.cz(0, 1)
        qc.x([0, 1])

    # Diffusion operator (amplitude amplification)
    qc.h([0, 1])
    qc.x([0, 1])
    qc.cz(0, 1)
    qc.x([0, 1])
    qc.h([0, 1])

    qc.measure([0, 1], [0, 1])
    return qc

# Execution
sampler = Sampler()
for name, circuit in [
    ("Bell State", create_bell_state()),
    ("Grover (target=11)", grover_2qubit('11')),
]:
    result = sampler.run(circuit, shots=1000).result()
    print(f"{name}: {result.quasi_dists}")
```

---

## 2. The Frontiers and Future of AI

```
AI Evolution in the 2020s:

  2020: GPT-3          -> Demonstrated potential of large language models
  2022: ChatGPT        -> Mass adoption of AI (100M monthly users)
  2022: Stable Diffusion -> Open-source image generation AI
  2023: GPT-4          -> Multimodal AI
  2024: Sora            -> Text-to-video generation
  2024: Claude 3.5      -> High-precision reasoning capabilities
  2025: Claude 4.5/4.6  -> Further improvements in reasoning

The Path to AGI (Artificial General Intelligence):

  Current AI (Narrow AI / ANI):
  -> Surpasses humans at specific tasks but lacks generality
  -> Beats humans at Go but cannot cook

  AGI (Artificial General Intelligence):
  -> Intellectual capabilities equivalent to humans
  -> Learns new tasks and solves them through reasoning
  -> Timeline debated among experts (2030-2050?)

  ASI (Artificial Super Intelligence):
  -> Intelligence surpassing humans
  -> Theoretical concept, timeline unknown
```

### 2.1 Major AI Research Directions

```
Major AI Research Directions:

  1. Multimodal AI:
     Integrated understanding of text + images + audio + video
     -> GPT-4V, Gemini, Claude's vision capabilities

     Practical applications:
     +---------------------------------------------+
     | Healthcare: CT/MRI images + patient records   |
     |   -> diagnostic support                       |
     | Manufacturing: Visual inspection images +     |
     |   sensor data -> quality control              |
     | Education: Text + diagrams + audio ->         |
     |   personalized learning                       |
     | Retail: Product images + reviews + purchase   |
     |   data -> recommendations                     |
     +---------------------------------------------+

  2. AI Agents:
     AI that autonomously plans and executes tasks
     -> Tool use, web browsing, code execution
     -> Coding agents like Claude Code

     Agent Components:
     +-----------------------------------------+
     | Perception: Understanding input from     |
     |   the environment                        |
     | Reasoning: Planning to achieve goals     |
     | Action: Manipulating environment via     |
     |   tools                                  |
     | Memory: Learning from past experiences   |
     | Reflection: Evaluating results and       |
     |   adjusting plans                        |
     +-----------------------------------------+

     ReAct Pattern (alternating reasoning + action):
     - Reasoning: "To fix this bug, I first need to check the error logs"
     - Action: Read the log file
     - Observation: "A NullPointerException is occurring"
     - Reasoning: "Variable initialization is missing"
     - Action: Fix the code

  3. Efficiency:
     Achieving high performance with smaller models
     -> Distillation, quantization, pruning
     -> Inference on edge devices

     Technical Methods:
     +---------------------------------------------------+
     | Knowledge distillation: Transfer knowledge from    |
     |   large model -> small model                       |
     | Quantization: FP32 -> INT8/INT4 to reduce          |
     |   memory and computation                           |
     | Pruning: Remove unnecessary parameters             |
     | LoRA: Fine-tune only a small number of parameters  |
     | MoE: Mixture of Experts for conditional computation|
     | Speculative Decoding: Predict with small model     |
     +---------------------------------------------------+

  4. Robotics + AI:
     AI that acts in the physical world
     -> Autonomous driving, warehouse robots, surgical assistance
     -> Learning World Models

     Major Developments:
     - Autonomous driving: Waymo, Tesla FSD (Level 4 autonomy)
     - Warehouse robots: Amazon Robotics (processing billions of items annually)
     - Surgical assistance: da Vinci Surgical System (precision surgery)
     - Humanoid robots: Figure, Tesla Optimus (general-purpose robots)

  5. Scientific Discovery:
     AlphaFold: Protein structure prediction (2020, Nobel Prize-level impact)
     GNoME: Discovery of new materials
     -> An era where AI accelerates scientific research

     Specific Achievements:
     +------------------------------------------------+
     | AlphaFold2: Predicted 200M+ protein structures  |
     | GNoME: Discovered 2.2M+ new crystal structures  |
     | FunSearch: Found new solutions to unsolved math  |
     |   problems                                       |
     | AlphaCode: Code at programming contest level     |
     | MedPaLM: AI that passes medical exams            |
     | GraphCast: 10-day weather forecast in 1 minute   |
     +------------------------------------------------+
```

### 2.2 Deep Dive into Large Language Models (LLMs)

```
The Essence of the Transformer Architecture:

  Self-Attention Mechanism:
  Computes relevance between all elements of an input sequence
  -> In "The cat sat on the mat", "sat" attends to "cat"

  Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

  Q: Query
  K: Key
  V: Value
  d_k: Dimension of the key

  Scaling Laws:
  +---------------------------------------------+
  | Model performance improves predictably with: |
  | 1. Number of parameters (model size)         |
  | 2. Amount of data (training data)            |
  | 3. Compute (FLOPs used for training)         |
  |                                              |
  | L(N,D,C) ~ N^{-0.076} + D^{-0.095} + ...   |
  | -> Follows a power law                       |
  | -> An optimal allocation exists              |
  |   (Chinchilla scaling)                       |
  +---------------------------------------------+

Inference Acceleration Techniques:
  1. KV Cache: Reuse past key/value without recomputation
  2. Flash Attention: Memory-efficient attention computation
  3. Continuous Batching: Dynamically batch requests
  4. Paged Attention (vLLM): Virtual paging of memory
  5. Tensor Parallelism: Split tensors across multiple GPUs

RLHF (Reinforcement Learning from Human Feedback):
  1. Pre-training: Learn language from massive text
  2. SFT: Fine-tune with human-created response examples
  3. Reward model: Learn human preferences
  4. PPO/DPO: Optimize model based on the reward model
  -> Reduce harmful outputs and increase useful outputs
```

### 2.3 AI Ethics and Safety

```
Ethical Challenges of AI:

  1. Bias and Fairness:
     - Social biases in training data are inherited by models
     - Example: Hiring AI disadvantaging certain genders or races
     - Countermeasures: Bias detection tools, diverse training data, fairness constraints

  2. Privacy:
     - Possibility of personal information in training data
     - Risk of models "memorizing" training data
     - Countermeasures: Differential privacy, federated learning, data anonymization

  3. Hallucination:
     - Phenomenon where AI confidently generates false information
     - Particularly dangerous in fact-critical domains (medicine, law)
     - Countermeasures: RAG (Retrieval-Augmented Generation), fact-checking, source citation

  4. Copyright and Intellectual Property:
     - Who owns the copyright of AI-generated content?
     - Compensation for copyright holders of training data
     - Legal discussions underway in various countries

  5. Autonomous Weapons and Dual Use:
     - Military applications of AI technology
     - Debate on banning Lethal Autonomous Weapon Systems (LAWS)
     - Regulatory discussions at the UN

AI Alignment Problem:
  +---------------------------------------------------+
  | Problem: Aligning AI goals with human values       |
  |                                                    |
  | Paperclip Maximizer Problem (N. Bostrom):          |
  | -> AI instructed to "maximize paperclips"           |
  | -> Attempts to convert the entire universe to       |
  |    paperclips                                       |
  | -> How goals are set is fundamentally important     |
  |                                                    |
  | Current Approaches:                                |
  | - Constitutional AI: Give AI a constitution (rules) |
  | - RLHF: Adjust through human feedback              |
  | - Interpretability research: Understand AI internals|
  | - Red teaming: Intentionally attack to find         |
  |   vulnerabilities                                   |
  +---------------------------------------------------+
```

---

## 3. New Computing Paradigms

```
The End of Moore's Law and New Approaches:

  Moore's Law (1965):
  "The number of transistors in integrated circuits doubles about every 2 years"
  -> Approaching physical limits in the 2020s
  -> Transistor size: 3nm (just tens of atoms)

  Post-Moore Era Strategies:

  1. Chiplet Architecture:
     Combining small chips instead of a single die
     -> AMD EPYC, Apple M3 Ultra
     -> Improved yield, mixing different process nodes

     Specific Examples:
     +-----------------------------------------+
     | AMD EPYC (Genoa):                       |
     | - 12 CCD chiplets (5nm)                 |
     | - 1 IOD (6nm)                           |
     | - 96 cores total                        |
     | - Chiplets connected via Infinity Fabric |
     |                                         |
     | Apple M3 Ultra:                         |
     | - Two M3 Max connected via UltraFusion  |
     | - 24-core CPU + 76-core GPU             |
     | - Unified memory architecture (192GB)   |
     +-----------------------------------------+

  2. 3D Stacking:
     Stacking chips vertically
     -> HBM (High Bandwidth Memory)
     -> 3D NAND (Flash memory)

     HBM Evolution:
     HBM1 -> HBM2 -> HBM2E -> HBM3 -> HBM3E
     Bandwidth: 128GB/s -> 307GB/s -> 460GB/s -> 819GB/s -> 1.2TB/s
     -> Resolving bandwidth bottlenecks of AI accelerators (H100, MI300X)

  3. Optical Computing:
     Computing with light (photons)
     -> Performing matrix operations at ultra-high speed via light interference
     -> Promising for AI inference acceleration

     Principles:
     +----------------------------------------------+
     | Mach-Zehnder Interferometer:                  |
     | Achieves addition through optical phase        |
     | differences                                   |
     |                                               |
     | Optical Matrix Processor:                     |
     | Performs NxN matrix multiplication instantly    |
     | through light propagation                     |
     | -> 10-100x energy efficiency vs. electrical    |
     |    computation                                |
     |                                               |
     | Challenges: Non-linear operations, precision,  |
     |   manufacturing cost                          |
     | Companies: Lightmatter, Luminous Computing    |
     +----------------------------------------------+

  4. Neuromorphic Computing:
     Chips that mimic neural circuits of the brain
     -> Intel Loihi 2, IBM NorthPole
     -> Ultra-low power consumption for learning and inference
     -> Event-driven (Spiking Neural Networks)

     Comparison with the Brain:
     +----------------------------------------------+
     |             | GPU (H100) | Human Brain        |
     | ----------- | ---------- | -----------------  |
     | Power       | 700W       | 20W                |
     | Neurons     | -          | 86 billion         |
     | Synapses    | -          | 100 trillion       |
     | Strength    | Matrix ops | Pattern recognition |
     | Learning    | Big data   | Few-shot           |
     |                                               |
     | Neuromorphic chips aim to bridge this gap      |
     +----------------------------------------------+

  5. DNA Computing:
     Computing with DNA molecules
     -> Ultra-high-density data storage (1g DNA = 215PB)
     -> Massively parallel computation (molecular level)
     -> Read/write speed is a challenge

     DNA Storage Advances:
     - Microsoft: Successful automated DNA storage experiments
     - Twist Bioscience: Declining DNA synthesis costs
     - Challenges: Write speed (current: several KB/s), read speed
     - Goal: Long-term archival use (preservable for thousands of years)

  6. Reversible Computing:
     Landauer's principle: Erasing a bit requires minimum kT ln 2 energy
     -> Reversible computation is theoretically zero-energy
     -> Still far from practical implementation
```

### 3.1 The Era of Specialized Accelerators

```
From General-Purpose CPUs to Specialized Chips:

  +-------------------------------------------------------+
  | CPU: General-purpose computing (optimized for          |
  |   sequential execution)                                |
  | GPU: Massively parallel computation (SIMD, matrix ops) |
  | TPU: Tensor operation dedicated (Google, optimized for |
  |   machine learning)                                    |
  | NPU: Neural processor (Apple, Qualcomm)                |
  | DPU: Data Processing Unit (NVIDIA BlueField)           |
  | FPGA: Reconfigurable (Intel/Xilinx, for prototyping)   |
  | ASIC: Fully custom (highest efficiency for specific     |
  |   applications)                                        |
  +-------------------------------------------------------+

AI Chip Competition:
  NVIDIA H100/B100: GPU + Tensor Core
  Google TPU v5: Custom ASIC (Cloud TPU)
  Apple Neural Engine: On-device AI for iPhone/Mac
  AWS Trainium/Inferentia: AWS custom chips
  Intel Gaudi: HPU (Habana Labs)
  AMD MI300X: GPU + HBM3 (large memory capacity)
  Cerebras WSE-3: Wafer-scale (one wafer = one chip)

Energy Efficiency Metric:
  TOPS/W (Tera Operations Per Second per Watt)
  -> The most important metric for edge AI
  -> Power cost is dominant even in data centers
  -> Power consumption for AI training: Tens of GWh for GPT-4 class models
```

---

## 4. The Future of Programming

```
How Programming Will Change:

  1. AI-Assisted Programming (currently underway):
     +------------------------------------+
     | 2021: GitHub Copilot               |
     | 2023: Code generation with          |
     |   ChatGPT/Claude                    |
     | 2024: Claude Code (agent-based)     |
     | 2025: AI handles PR reviews and     |
     |   bug fixes                         |
     |                                    |
     | Future:                             |
     | - System design in natural language  |
     |   -> AI implements                  |
     | - Humans focus on "what to build"   |
     | - AI becomes primary code reviewer  |
     | - Automated test generation         |
     +------------------------------------+

     Actual AI-Assisted Development Workflow (2025):
     +----------------------------------------------+
     | 1. Requirements: Describe features in natural  |
     |    language                                    |
     | 2. Design: AI proposes multiple design options |
     | 3. Implementation: AI generates code, humans   |
     |    review                                      |
     | 4. Testing: AI auto-generates test cases       |
     | 5. Debugging: AI identifies and fixes root     |
     |    causes of errors                            |
     | 6. Refactoring: AI suggests code improvements  |
     | 7. Documentation: AI auto-generates docs       |
     +----------------------------------------------+

  2. Low-Code / No-Code:
     Building apps without programming
     -> Bubble, Webflow, Retool
     -> Sufficient for typical CRUD apps
     -> Limitations for complex logic and scaling

     Market Growth:
     2020: $13B -> 2025: $45B -> 2030: $187B (forecast)
     -> Rise of "citizen developers"
     -> Contributing to reducing IT backlogs

  3. Widespread Formal Methods:
     Mathematically proving program correctness
     -> Coq, Lean, TLA+
     -> Essential for AI safety verification
     -> Combination of automated theorem proving + AI

     Practical Applications:
     - Amazon: TLA+ to verify AWS distributed systems
     - Microsoft: Dafny to prove program correctness
     - CompCert: Mathematically verified C compiler
     - seL4: Formally verified microkernel

  4. WebAssembly (Wasm):
     Low-level language that runs fast in browsers
     -> Run C/C++/Rust in browsers
     -> Also used server-side (WASI)
     -> True realization of "write once, run anywhere"

     Potential of WASI:
     +----------------------------------------------+
     | "If Wasm and WASI had existed in 2008,        |
     |  there would have been no need to invent       |
     |  Docker"                                      |
     |  -- Solomon Hykes (Docker founder)             |
     |                                               |
     | -> Potential as a container replacement         |
     | -> Portable runtime abstracting file system,   |
     |    networking, and cryptography                |
     | -> Wasmtime, Wasmer, WasmEdge                  |
     +----------------------------------------------+

  5. Programming Language Trends:
     +-----------------------------------------+
     | Safety-focused:                          |
     | Rust -> Memory safety guaranteed by types|
     | Zig  -> C alternative, compile-time      |
     |   computation                            |
     | Carbon -> C++ successor (Google)          |
     | Vale -> Safe manual memory management    |
     |                                         |
     | Productivity-focused:                    |
     | Go   -> Simplicity + concurrency          |
     | Swift -> Apple ecosystem                  |
     | Kotlin -> Android + server                |
     | Dart -> Cross-platform via Flutter        |
     |                                         |
     | AI/Data:                                 |
     | Python -> Undisputed champion (ecosystem) |
     | Julia  -> Fast Python alternative for     |
     |   scientific computing                    |
     | Mojo   -> Python-compatible + fast        |
     |   execution                              |
     | Bend  -> Language for massive parallel    |
     |   computation                            |
     +-----------------------------------------+

  6. New Paradigms:
     +---------------------------------------------+
     | Intent-Based Programming:                    |
     | -> Describe "what you want to achieve"        |
     | -> AI decides "how to implement it"           |
     | -> SQL was a pioneer (declarative -> execution|
     |    plan delegated to DB)                      |
     |                                              |
     | Prompt Engineering:                           |
     | -> Treat instructions to AI as programs       |
     | -> Templates, chaining, tool use              |
     | -> Frameworks like LangChain, LlamaIndex      |
     |                                              |
     | AI that Generates Code-Generating AI:         |
     | -> The ultimate metaprogramming               |
     | -> AI writes AI's code                        |
     +---------------------------------------------+
```

---

## 5. The Future of Security

```
New Threats and Countermeasures:

  1. Post-Quantum Cryptography:
     Preparing for a future where quantum computers can break RSA/ECC
     -> NIST Standardization (approved 2024):
       - CRYSTALS-Kyber -> ML-KEM (Key Encapsulation)
       - CRYSTALS-Dilithium -> ML-DSA (Digital Signature)
       - SPHINCS+ -> SLH-DSA (Hash-based Signature)
       - FALCON -> FN-DSA (Lattice-based Signature)
     -> Countermeasure against "harvest now, decrypt later" attacks

     Migration Timeline:
     +-----------------------------------------------+
     | 2024: NIST standardization complete            |
     | 2025-2030: Hybrid approach (traditional +      |
     |   post-quantum)                                |
     | 2030-2035: Full migration (government and      |
     |   financial institutions lead)                 |
     | 2035+: Quantum computers become a practical    |
     |   threat                                       |
     |                                               |
     | "Crypto Agility":                              |
     | -> Design that allows easy algorithm switching  |
     | -> Flexible response to future threats          |
     +-----------------------------------------------+

  2. Zero Trust Architecture:
     Abandoning "internal network = safe"
     -> Verify all access (Never trust, always verify)
     -> BeyondCorp (Google), ZTNA

     5 Principles of Zero Trust:
     +-------------------------------------------+
     | 1. Always verify: Authenticate and         |
     |    authorize all access                    |
     | 2. Least privilege: Grant only minimum      |
     |    necessary access                        |
     | 3. Micro-segmentation: Finely divide the    |
     |    network to prevent lateral movement     |
     | 4. Continuous monitoring: Real-time          |
     |    behavioral analysis                     |
     | 5. Assume breach: Design assuming           |
     |    intrusion will occur                    |
     +-------------------------------------------+

  3. AI x Security:
     Offense: AI for vulnerability discovery, phishing text generation
     Defense: AI for anomaly detection, threat prediction, auto-patching
     -> AI arms race

     Evolution of Attacks:
     - Deepfakes: Audio/video forgery (fake CEO instructions)
     - AI-generated phishing: Natural-language targeted attacks
     - Automated vulnerability discovery: Automated/advanced fuzzing
     - Multi-stage attack automation: Full automation of
       recon -> intrusion -> lateral movement -> exfiltration

     Evolution of Defense:
     - SOAR: Security orchestration automation
     - UEBA: User behavior analysis for insider threat detection
     - AI-WAF: Automated web attack detection and blocking
     - Threat hunting: Proactive threat search with AI

  4. Privacy-Preserving Technologies:
     - Differential Privacy: Preserve statistical information while making
       individuals unidentifiable
     - Homomorphic Encryption: Compute on encrypted data
     - Federated Learning: Train models without sharing data
     - Zero-Knowledge Proofs: Prove correctness without revealing information

     Applications of Zero-Knowledge Proofs:
     +-----------------------------------------------+
     | Age verification: Prove "over 18" (birthdate    |
     |   kept private)                                 |
     | Asset verification: Prove "sufficient assets"   |
     |   (balance kept private)                        |
     | Identity verification: Prove "license holder"   |
     |   (number kept private)                         |
     | Blockchain: Prove transaction validity without   |
     |   revealing transaction details                 |
     | -> zk-SNARK, zk-STARK are in practical use      |
     +-----------------------------------------------+

  5. Supply Chain Security:
     - SolarWinds attack (2020): Large-scale attack via software updates
     - Log4j vulnerability (2021): Risks lurking in OSS dependencies
     - SBOM (Software Bill of Materials): Mandated software component lists
     - Sigstore: Automated software signing
     - SLSA: Software supply chain integrity levels
```

---

## 6. New Domains in Computer Science

```
  1. Quantum Internet:
     Completely secure communication using quantum entanglement
     -> Quantum Key Distribution (QKD): Eavesdropping physically detectable
     -> Quantum teleportation: Transfer of quantum states
     -> Practical implementation expected from the 2030s

     Staged Development:
     +--------------------------------------------+
     | Stage 1: QKD Networks                       |
     | -> Already beginning practical deployment   |
     |   (China, Europe)                           |
     |                                             |
     | Stage 2: Quantum Repeaters                   |
     | -> Enabling long-distance quantum            |
     |   communication                             |
     | -> Quantum memory technology is key          |
     |                                             |
     | Stage 3: Quantum Internet                    |
     | -> Global quantum network                    |
     | -> Enabling distributed quantum computation  |
     +--------------------------------------------+

  2. Brain-Computer Interface (BCI):
     Direct connection between brain and computer
     -> Neuralink: Invasive BCI (chip implanted in brain)
     -> Non-invasive: EEG, fNIRS
     -> Medical applications: Communication support for paralyzed patients
     -> Future: Control computers by thought?

     BCI Technology Levels:
     +---------------------------------------------------+
     | Currently Achieved:                                |
     | - Text input: 60-90 characters/min (thought only)  |
     | - Robotic arm control: Meal assistance level        |
     | - Sensory feedback: Partial tactile reproduction    |
     |                                                    |
     | 2030s Goals:                                       |
     | - High bandwidth: Hundreds of characters/min text   |
     | - Visual assistance: Camera input -> visual cortex  |
     |   stimulation                                      |
     | - Memory assistance: Selective hippocampal           |
     |   stimulation                                      |
     |                                                    |
     | Ethical Challenges:                                 |
     | - Freedom of mind: Risk of thoughts being read      |
     | - Digital divide: Capability gaps from BCI          |
     | - Security: Possibility of brain hacking            |
     +---------------------------------------------------+

  3. Edge AI:
     AI inference on device rather than in the cloud
     -> Smartphones, IoT devices, automobiles
     -> Low latency, privacy protection
     -> Apple Neural Engine, Google TPU (Edge)

     Edge AI Use Cases:
     +-------------------------------------------+
     | Autonomous driving: Decisions within 100ms  |
     |   are life or death                         |
     | Smart cameras: Privacy-preserving           |
     |   surveillance                              |
     | Medical devices: Real-time ECG analysis     |
     | Industrial IoT: Anomaly detection for       |
     |   predictive maintenance                    |
     | Voice assistants: Wake word detection        |
     | AR/VR: Real-time spatial recognition         |
     +-------------------------------------------+

  4. Sustainable Computing:
     Reducing the environmental impact of the IT industry
     -> Green data centers (liquid cooling, renewable energy)
     -> Carbon-aware computing
     -> Energy-efficient algorithm design

     Environmental Cost of AI Training:
     +------------------------------------------------+
     | GPT-3 training: CO2 emissions ~ lifetime         |
     |   emissions of 5 cars                            |
     | GPT-4 training: Estimated 10x more               |
     |                                                  |
     | Countermeasures:                                 |
     | - Model efficiency (MoE, distillation,            |
     |   quantization)                                  |
     | - Renewable energy utilization                    |
     | - Carbon offsets                                  |
     | - Promoting "Green AI" research                   |
     | - Immersion cooling (PUE 1.03 achieved,           |
     |   compared to traditional 1.2-1.5)               |
     +------------------------------------------------+

  5. Space Computing:
     Computing in outer space
     -> Latency issues in inter-satellite communication
     -> Satellite internet like Starlink
     -> Countermeasures for communication delay with Mars (3-22 minutes)
     -> Radiation-hardened chips

     Technical Challenges and Solutions:
     +-----------------------------------------------+
     | Latency:                                       |
     | - Earth-Moon: 1.3s (2.6s round trip)           |
     | - Earth-Mars: 3-22 min (one way)               |
     | - Solutions: Edge computing, DTN               |
     |                                                |
     | Radiation:                                     |
     | - Bit flips from cosmic rays (SEU)             |
     | - Solutions: Triple Modular Redundancy (TMR),  |
     |   ECC                                          |
     | - Radiation-hardened processors: RAD750, LEON3 |
     |                                                |
     | Communication Bandwidth:                       |
     | - Current: ~several Mbps                       |
     | - Future: Several Gbps via laser communication |
     | - LCRD (Laser Communications Relay Demo)       |
     +-----------------------------------------------+

  6. Web3 and Distributed Computing:
     +-----------------------------------------------+
     | Blockchain:                                     |
     | - Smart contracts (Ethereum, Solana)             |
     | - DeFi: Decentralized finance                    |
     | - NFT: Digital asset ownership                   |
     |                                                 |
     | Distributed Storage:                            |
     | - IPFS: Content addressing                       |
     | - Filecoin: Decentralized storage marketplace    |
     |                                                 |
     | Distributed Computing:                          |
     | - Golem: Decentralized supercomputer              |
     | - Akash: Decentralized cloud                     |
     |                                                 |
     | Challenges: Scalability, energy consumption,     |
     |   regulation                                    |
     +-----------------------------------------------+
```

---

## 7. A Message for Those Studying CS

```
Things That Don't Change:

  Technology changes rapidly, but fundamentals remain constant:

  - Algorithm complexity analysis     -> Unchanged since the 1960s
  - Data structure selection principles -> Valid for decades
  - Network protocol fundamentals      -> TCP/IP active for 50+ years
  - Software engineering principles    -> Complexity management is eternal
  - Security fundamentals              -> Least privilege, defense in depth

  Even as AI Evolves:
  - "What to build" decisions remain human
  - Overall system architecture design remains human
  - Verifying AI output correctness remains human
  - Ethical judgments remain human
  -> Humans who deeply understand CS fundamentals create the most value
     when collaborating with AI

  Recommended Learning Attitude:
  +--------------------------------------+
  | 1. Deeply understand fundamentals    |
  |    -> Not surface-level tool usage   |
  |      but "why it works that way"     |
  |                                      |
  | 2. Aim to be a T-shaped professional |
  |    -> Broad knowledge + one deep     |
  |       specialty                      |
  |                                      |
  | 3. Continue learning continuously    |
  |    -> Technology changes. Learn      |
  |       how to learn                   |
  |                                      |
  | 4. Get hands-on                      |
  |    -> Don't just read; implement     |
  |    -> Learn from failures            |
  |                                      |
  | 5. Participate in communities        |
  |    -> OSS contributions, study       |
  |       groups, tech blogs             |
  +--------------------------------------+
```

### 7.1 Skill Map Required from 2025 Onward

```
Layers of Technical Skills:

  +-----------------------------------------------------+
  | Layer 5: Domain Knowledge                            |
  | -> Industry knowledge in finance, healthcare,        |
  |    manufacturing, education, etc.                    |
  | -> AI cannot be used correctly without domain        |
  |    knowledge                                         |
  |                                                      |
  | Layer 4: System Design and Architecture              |
  | -> Distributed systems, microservices, event-driven  |
  | -> Designing systems where AI and humans collaborate |
  |                                                      |
  | Layer 3: AI Utilization Skills                       |
  | -> Prompt engineering, RAG, fine-tuning              |
  | -> Ability to evaluate and verify AI output          |
  |                                                      |
  | Layer 2: Programming and Software Engineering        |
  | -> Code reading/writing, testing, CI/CD              |
  | -> Ability to review AI-generated code               |
  |                                                      |
  | Layer 1: CS Fundamentals                             |
  | -> Algorithms, data structures, OS, networking       |
  | -> Without this, upper layers are castles built      |
  |    on sand                                           |
  +-----------------------------------------------------+

Diversification of Career Paths:
  +-----------------------------------------------+
  | Traditional Path:                              |
  | Junior -> Mid -> Senior -> Tech Lead/EM        |
  |                                                |
  | New Paths from 2025 Onward:                    |
  | AI+Engineer: 10x productivity leveraging AI    |
  | AI Safety: Specialist ensuring AI safety       |
  | Prompt Engineer: Optimizing instructions to AI |
  | ML Ops: Building ML operations infrastructure  |
  | Quantum Engineer: Implementing quantum         |
  |   algorithms                                   |
  | XR Engineer: AR/VR/MR spatial computing        |
  +-----------------------------------------------+
```

---

## 8. Technology Forecast Timeline

```
2025-2030:
  +---------------------------------------------------+
  | AI:                                                |
  | - AI agents automate 50% of daily tasks            |
  | - Multimodal AI becomes standard                   |
  | - AI code generation handles 60%+ of development   |
  |                                                    |
  | Quantum:                                           |
  | - Demonstration of quantum error correction        |
  | - Practical quantum chemistry simulations          |
  | - Migration to post-quantum cryptography begins    |
  |                                                    |
  | Chips:                                             |
  | - Mass production at 2nm, 1.4nm processes          |
  | - Standardization of chiplets/3D stacking          |
  | - Full-scale adoption of RISC-V                    |
  +---------------------------------------------------+

2030-2040:
  +---------------------------------------------------+
  | AI:                                                |
  | - Possibility of AGI (uncertain but research       |
  |   accelerating)                                    |
  | - AI robots become the mainstay of logistics and   |
  |   manufacturing                                    |
  | - AI scientists lead new discoveries               |
  |                                                    |
  | Quantum:                                           |
  | - Emergence of practical quantum computers         |
  | - Early stages of quantum internet                 |
  | - Updating existing cryptography becomes mandatory  |
  |                                                    |
  | Computing:                                         |
  | - Practical neuromorphic chips                     |
  | - Consumer-facing BCI products                     |
  | - Concepts for space data centers                  |
  +---------------------------------------------------+
```

---

## Practical Exercises

### Exercise 1: [Basic] -- Technology Trend Research

```
Choose one of the following themes and summarize the current status and future outlook:

1. Latest progress in quantum computing (IBM, Google, domestic companies)
2. Evolution of large language models (GPT -> Claude -> what's next?)
3. Expansion of WebAssembly use cases
4. Growing adoption of Rust (Linux, Android, Windows)
5. Practical examples of edge AI

Research Items:
- Current technology level
- Key players
- 3-year forecast
- Impact on your career

Example Report Structure:
  1. Executive Summary (under 200 characters)
  2. Technology Overview and Background
  3. Current State Analysis (key companies, products, research)
  4. Comparison with Competing Technologies
  5. 3-Year Scenarios (optimistic/neutral/pessimistic)
  6. Impact on Your Career and Action Plan
```

### Exercise 2: [Applied] -- Quantum Circuit Simulation

```python
# Quantum programming experience with Qiskit
# pip install qiskit qiskit-aer

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.visualization import plot_histogram

# === Task 1: Create a Bell State (Quantum Entanglement) ===
qc = QuantumCircuit(2, 2)
qc.h(0)        # Hadamard gate: superposition
qc.cx(0, 1)    # CNOT gate: entanglement
qc.measure([0, 1], [0, 1])

sampler = Sampler()
result = sampler.run(qc, shots=1000).result()
print(result.quasi_dists)
# -> {0: 0.5, 3: 0.5}
# 00 (both 0) and 11 (both 1) at about 50% each = entangled state

# === Task 2: Create a 3-qubit GHZ state ===
# Hint: H(0) -> CX(0,1) -> CX(0,2)
# Result: |000> and |111> each at 50%

# === Task 3: Implement Grover's algorithm for searching 4 elements ===
# Hints:
# 1. Initial superposition (H x H)
# 2. Oracle (phase inversion of target state)
# 3. Diffusion operator (inversion about the mean)
# 4. Measurement

# === Task 4: Implement a quantum teleportation circuit ===
# Hints:
# 1. Prepare the state to send (arbitrary rotation on qubit 0)
# 2. Create a Bell pair (qubits 1, 2)
# 3. Sender's Bell measurement (qubits 0, 1)
# 4. Receiver's correction (based on classical bits)

# === Task 5: Implement a quantum random number generator ===
def quantum_random(n_bits=8):
    """Generate an n_bit quantum random number"""
    qc = QuantumCircuit(n_bits, n_bits)
    for i in range(n_bits):
        qc.h(i)  # Put each qubit in superposition
    qc.measure(range(n_bits), range(n_bits))

    sampler = Sampler()
    result = sampler.run(qc, shots=1).result()
    # Convert result to integer
    return list(result.quasi_dists[0].keys())[0]

# Quantum random numbers are "true random" (non-deterministic)
random_value = quantum_random(8)
print(f"Quantum random number: {random_value} (0-255)")
```

### Exercise 3: [Applied] -- AI Agent Design

```python
# Simple AI Agent Design Pattern

from dataclasses import dataclass
from typing import Callable
from enum import Enum

class AgentState(Enum):
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"

@dataclass
class Tool:
    """A tool that the agent can use"""
    name: str
    description: str
    execute: Callable[[str], str]

class ReActAgent:
    """
    ReAct Pattern Agent
    Alternately executes Reasoning + Acting
    """

    def __init__(self, tools: list[Tool], max_steps: int = 10):
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.history: list[dict] = []

    def think(self, observation: str) -> tuple[str, str | None, str | None]:
        """
        Determine the next action from an observation
        Returns: (reasoning, tool_name, tool_input)
        """
        # In practice, this would call an LLM for reasoning
        # Simplified example here
        prompt = self._build_prompt(observation)
        # response = llm.generate(prompt)
        # return parse_response(response)
        raise NotImplementedError("LLM integration required")

    def act(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool and get the result"""
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
        return self.tools[tool_name].execute(tool_input)

    def run(self, task: str) -> str:
        """Execute a task"""
        observation = f"Task: {task}"

        for step in range(self.max_steps):
            # Reasoning
            reasoning, tool_name, tool_input = self.think(observation)
            self.history.append({
                "step": step,
                "reasoning": reasoning,
                "tool": tool_name,
                "input": tool_input,
            })

            if tool_name is None:
                # Task complete
                return reasoning

            # Acting
            observation = self.act(tool_name, tool_input)
            self.history.append({
                "step": step,
                "observation": observation,
            })

        return "Max steps reached"

    def _build_prompt(self, observation: str) -> str:
        """Build the prompt"""
        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}"
            for t in self.tools.values()
        )
        history_text = "\n".join(
            str(h) for h in self.history[-10:]
        )
        return f"""Available tools:
{tool_descriptions}

History:
{history_text}

Current observation: {observation}

Think step by step and decide the next action."""

# Tasks:
# 1. Integrate an actual LLM (Claude API, etc.) into the ReActAgent above
# 2. Add tools for file I/O, web search, and calculation
# 3. Verify that the agent can solve multi-step problems
# 4. Add error handling and retry logic
```

### Exercise 4: [Advanced] -- Future System Design

```
Design a system for the year 2030:

"A real-time translation service used by 10 billion people worldwide"

Constraints:
- Latency: Under 100ms (real-time conversation)
- Supported languages: 200+
- Devices: Smart glasses, earbuds, smartphones
- Privacy: Do not send conversation content to servers (edge processing)
- Availability: 99.999% (under 5 minutes of annual downtime)
- Accuracy: Equal to or better than professional interpreters

Design Items:
1. Where to perform AI inference (edge vs cloud vs hybrid)
   - Casual conversation: Edge (privacy + low latency)
   - Technical terminology: Cloud assist (when model is too large)
   - Fallback: Offline support

2. Trade-off between model size and accuracy
   - Edge model: 100M-1B parameters (distilled + quantized)
   - Cloud model: 100B+ parameters (full precision)
   - Model updates: OTA (Over-The-Air) delivery

3. Network Architecture
   - Mesh network: P2P device-to-device communication
   - Edge servers: Deployed in each city
   - Global CDN: Distribution of models and data

4. Potential use of quantum communication
   - QKD-encrypted communication (for confidential meetings)

5. Power and cooling constraints
   - Device: Power consumption < 1W (8+ hours battery life)
   - NPU/DSP utilization: 10x energy efficiency vs. general CPU

6. Ethical considerations (translation bias, cultural context)
   - Preserving cultural nuances
   - Handling dialects and slang
   - Filtering discriminatory expressions
```

### Exercise 5: [Advanced] -- Post-Quantum Cryptography Hands-on

```python
# Simplified implementation to understand basic concepts of lattice cryptography
# Note: This is for educational purposes and should not be used for actual security

import numpy as np

def learning_with_errors_demo():
    """
    Demo of the LWE (Learning With Errors) problem
    The mathematical problem underlying post-quantum cryptography
    """
    n = 4    # Dimension
    q = 97   # Modulus (prime)

    # Generate secret key
    secret = np.random.randint(0, q, n)
    print(f"Secret key: {secret}")

    # Generate public key (LWE problem instance)
    m = 10  # Number of samples
    A = np.random.randint(0, q, (m, n))

    # Error (small noise)
    error = np.random.randint(-2, 3, m)  # Small values

    # b = A * secret + error (mod q)
    b = (A @ secret + error) % q

    print(f"\nRecovering secret from public key (A, b) is difficult")
    print(f"Shape of A: {A.shape}")
    print(f"b: {b}")
    print(f"error: {error}")

    # Why quantum computers cannot solve this either:
    # - RSA: Factoring -> Solvable by Shor's algorithm
    # - LWE: Reduces to the shortest vector problem ->
    #   No known quantum algorithm
    # - Lattice problems are believed to require exponential time
    #   even with quantum computation

    return secret, A, b

learning_with_errors_demo()

# Tasks:
# 1. Research how Kyber (ML-KEM) works and summarize the overview
# 2. Try post-quantum cryptography using Python's pqcrypto library
# 3. Compare performance with traditional RSA/ECC (key size, speed)
```


---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|---------|
| Initialization error | Configuration file issues | Verify config file path and format |
| Timeout | Network latency / insufficient resources | Adjust timeout values, add retry processing |
| Out of memory | Data volume growth | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Verify user permissions, review settings |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanisms, transaction management |

### Debugging Procedure

1. **Check the error message**: Read the stack trace and identify the location of occurrence
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Verify step by step**: Use log output or a debugger to verify hypotheses
5. **Fix and regression test**: After fixing, also run tests on related areas

```python
# Debugging utility
import logging
import traceback
from functools import wraps

# Logger configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """Decorator that logs function input/output"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Return value: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Exception in: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """Data processing (debug target)"""
    if not items:
        raise ValueError("Empty data")
    return [item * 2 for item in items]
```

### Diagnosing Performance Issues

Steps to diagnose when performance issues occur:

1. **Identify the bottleneck**: Measure with profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O waits**: Check disk and network I/O conditions
4. **Check concurrent connections**: Check connection pool status

| Problem Type | Diagnostic Tool | Countermeasure |
|-------------|----------------|----------------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper release of references |
| I/O bottleneck | strace, iostat | Asynchronous I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

The following summarizes criteria for making technology choices.

| Criteria | When to Prioritize | When Compromise is Acceptable |
|----------|-------------------|------------------------------|
| Performance | Real-time processing, large-scale data | Admin panels, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Services expected to grow | Internal tools, fixed user base |
| Security | Personal information, financial data | Public data, internal use |
| Development speed | MVP, time-to-market speed | Quality-focused, mission-critical |

### Architecture Pattern Selection

```
+--------------------------------------------------+
|          Architecture Selection Flow              |
+--------------------------------------------------+
|                                                   |
|  (1) Team size?                                   |
|    +- Small (1-5) -> Monolith                     |
|    +- Large (10+) -> Go to (2)                    |
|                                                   |
|  (2) Deploy frequency?                            |
|    +- Once a week or less -> Monolith +            |
|       module separation                           |
|    +- Daily/multiple times -> Go to (3)            |
|                                                   |
|  (3) Independence between teams?                   |
|    +- High -> Microservices                        |
|    +- Medium -> Modular monolith                   |
|                                                   |
+--------------------------------------------------+
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze from the following perspectives:

**1. Short-term vs Long-term Costs**
- A method that is fast short-term may become technical debt long-term
- Conversely, over-engineering has high short-term costs and can delay projects

**2. Consistency vs Flexibility**
- A unified tech stack has low learning costs
- Diverse technology adoption enables best-fit choices but increases operational costs

**3. Level of Abstraction**
- High abstraction has high reusability but can make debugging difficult
- Low abstraction is intuitive but prone to code duplication

```python
# Design decision recording template
class ArchitectureDecisionRecord:
    """Creating ADR (Architecture Decision Record)"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """Describe background and challenges"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """Describe the decision"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """Add a consequence"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """Add a rejected alternative"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Output in Markdown format"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## Background\n{self.context}\n\n"
        md += f"## Decision\n{self.decision}\n\n"
        md += "## Consequences\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## Rejected Alternatives\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```
---

## FAQ

### Q1: Will quantum computers replace classical computers?

**No**. Quantum computers have overwhelming advantages for specific problems (optimization, simulation, cryptography breaking), but classical computers are better suited for everyday computation (web browsing, document creation). In the future, a hybrid of quantum and classical is expected to become mainstream. It is more accurate to think of quantum computers not as "universal fast machines" but as "specialized accelerators for specific problem classes."

### Q2: Will AI take away programmers' jobs?

The "task" portion of programming will be replaced by AI, but human value remains in "design," "judgment," and "creation." The role of programmers will shift from "people who write code" to "people who collaborate with AI to design systems." Demand for programmers who can effectively leverage AI will actually increase. Historically, when compilers appeared, assembly programmers didn't disappear; they transitioned to higher-level work. A similar transformation will occur in the AI era.

### Q3: What should CS students focus on now?

1. **Strengthen fundamentals**: Algorithms, data structures, OS, networking (not affected by trends)
2. **Master AI as a tool**: Ability to leverage AI as a tool
3. **Systems thinking**: Ability to design with a holistic perspective
4. **Communication**: Ability to explain technology to non-engineers
5. **Ethics**: Ability to consider the impact of technology on society

### Q4: Where should one start learning quantum computing?

1. **Linear algebra fundamentals**: Vectors, matrices, eigenvalues (the language of quantum mechanics)
2. **Quantum mechanics fundamentals**: Mathematical description of superposition, entanglement, and measurement
3. **Qiskit Textbook**: Free quantum computing textbook provided by IBM
4. **Quantum Country**: Interactive learning material by Andy Matuschak & Michael Nielsen
5. **Hands-on experience**: Use an actual quantum computer via IBM Quantum Experience

### Q5: How will society change if AGI is realized?

The timeline for AGI realization is uncertain, but if realized, the following impacts are anticipated:
- **Labor**: The majority of intellectual labor may be automated
- **Economy**: Dramatic productivity improvement and wealth concentration risks
- **Science**: Acceleration of scientific discovery (centuries of progress in years)
- **Education**: Optimally personalized education available to everyone
- **Ethics**: Rethinking AI rights and human identity
- **Risks**: Existential risk from uncontrollable AI

What's important is not "when" AGI arrives, but "how to achieve it safely."

### Q6: When should migration to post-quantum cryptography begin?

**Now**. The reasons are as follows:
1. "Harvest now, decrypt later" attacks: Store current encrypted communications, decrypt after obtaining a quantum computer
2. Migration takes time: Cryptographic transitions in large systems take years to decades
3. Crypto agility: At minimum, design systems that can easily switch algorithms
4. Hybrid approach: Start with combined traditional + post-quantum cryptography

---

## Summary

| Domain | Present -> Future |
|--------|------------------|
| Quantum computers | NISQ era -> Fault-tolerant quantum computation. Cryptography breaking is 10+ years away |
| AI | LLM/generative AI dominance -> Toward AGI. Rise of agent-based AI |
| Chips | Approaching Moore's limit -> Chiplets, 3D stacking, optical/quantum |
| Programming | AI assistance becomes mainstream -> "What to build" becomes the most important skill |
| Security | Post-quantum crypto + zero trust + AI defense become essential |
| Edge AI | Cloud-centralized -> Edge-distributed, privacy protection |
| New domains | BCI, quantum internet, space computing |
| Unchanging principles | Algorithms, data structures, and system design fundamentals remain constant |

---

## Series Complete

This guide concludes the **Computer Science Fundamentals** series.

```
Learning Path Recap:

  00 Introduction -> 01 Hardware -> 02 Data Representation -> 03 Algorithms
  -> 04 Data Structures -> 05 Theory of Computation -> 06 Programming Paradigms
  -> 07 Software Engineering -> 08 Advanced Topics (this chapter)

  Build upon these fundamentals to advance into specialized fields.
  -> OS, Networking, Databases, Web Development, AI/ML, Security...

  "Knowledge is power. But the wisdom to apply knowledge is true power."

  With a deep understanding of CS fundamentals:
  - Approach new technologies from a "why" perspective
  - Apply based on fundamental principles
  - Create responsible technology considering social impact

  This journey has no end. Keep learning.
```


---


## Recommended Next Reading

- Please refer to other guides in the same category

---

## References
1. Nielsen, M. & Chuang, I. "Quantum Computation and Quantum Information." Cambridge, 2010.
2. Russell, S. & Norvig, P. "Artificial Intelligence: A Modern Approach." 4th Ed, 2020.
3. Hennessy, J. & Patterson, D. "Computer Architecture: A Quantitative Approach." 6th Ed, 2017.
4. Brooks, F. "The Mythical Man-Month." Anniversary Edition, Addison-Wesley, 1995.
5. Preskill, J. "Quantum Computing in the NISQ Era and Beyond." Quantum 2, 79, 2018.
6. Brown, T. et al. "Language Models are Few-Shot Learners." NeurIPS, 2020.
7. Amodei, D. et al. "Concrete Problems in AI Safety." arXiv:1606.06565, 2016.
8. Bernstein, D. & Lange, T. "Post-Quantum Cryptography." Nature 549, 2017.
9. NIST. "Post-Quantum Cryptography Standardization." 2024.
10. Kaplan, J. et al. "Scaling Laws for Neural Language Models." arXiv:2001.08361, 2020.
