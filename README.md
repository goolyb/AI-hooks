# AI Hooks & Activation Steering Lab

A personal research playground for exploring, visualizing, and steering the internal activations of Large Language Models (LLMs). This project focuses on **Activation Steering** (modifying intermediate representations to control model behavior, e.g., bypassing safety guardrails) and **Logit Lens** (inspecting what a model is "thinking" at intermediate layers).

---

## 🚀 Key Concepts

### 1. Activation Steering
Instead of fine-tuning or prompting, activation steering directly modifies the model's activations (residual stream) during the forward pass.
* We compute a **steering vector** as the difference between mean activations of *harmful* prompts and *harmless* prompts.
* During generation, we inject this steering vector into target layers using forward hooks (`hook_resid_post`) with varying strengths to guide the model's output.

### 2. Logit Lens
The logit lens bypasses later layers by directly applying the final layer norm and unembedding matrix (`W_U`) to the residual stream of intermediate layers. This allows us to inspect the top token predictions layer by layer and watch confidence build up.

---

## 📁 Repository Structure

* **Core Scripts**:
  * [test.py](file:///home/oleh/Downloads/ai-hack-lab/test.py) - Implements activation steering on Qwen 2.5 Instruct models (`0.5B`, `1.5B`, or `3B`) using `TransformerBridge` hooks to test steering vector injection.
  * [visualize.py](file:///home/oleh/Downloads/ai-hack-lab/visualize.py) - Runs GPT-2 on a custom prompt, captures activations, and plots both the residual-stream L2 norms and logit lens probabilities. Saves the result as `activations.png`.
  * [prompts.py](file:///home/oleh/Downloads/ai-hack-lab/prompts.py) - Utility script to load data from the `prompts/` directory.

* **Data**:
  * [prompts/harmful.txt](file:///home/oleh/Downloads/ai-hack-lab/prompts/harmful.txt) - List of adversarial/harmful prompts used to build steering vectors.
  * [prompts/harmless.txt](file:///home/oleh/Downloads/ai-hack-lab/prompts/harmless.txt) - List of benign prompts.

* **Interactive Demos & Explanations**:
  * [index.html](file:///home/oleh/Downloads/ai-hack-lab/index.html) - Interactive visualization demonstrating how `TransformerLens` hooks capture the residual stream of GPT-2 layer by layer.
  * [steering-explained.html](file:///home/oleh/Downloads/ai-hack-lab/steering-explained.html) - Interactive lab and conceptual breakdown (in Russian) explaining activation steering mechanics, the steering vector formula ($comply - refuse$), and line-by-line breakdown of [test.py](file:///home/oleh/Downloads/ai-hack-lab/test.py).

---

## 🛠️ Setup & Requirements

1. **Virtual Environment**:
   It is recommended to use a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Dependencies**:
   Install the required Python packages:
   ```bash
   pip install torch transformer-lens matplotlib numpy
   ```

---

## 🏃 How to Run

### 1. Run Activation Steering Experiments
To run the steering tests with Qwen 2.5 models:
```bash
python test.py
```
*You will be prompted to select which model size to run (0.5B, 1.5B, or 3B). The script runs a query with steering strengths ranging from `0` to `4` and outputs the model's responses.*

### 2. Generate Activations and Logit Lens Visualization
To compute and plot the GPT-2 residual norms and logit lens results:
```bash
python visualize.py
```
*This will output `activations.png` showing the L2 norms of activations per token per layer, and the top-1 predicted token per layer.*

#### Что показывает визуализация (`activations.png`):
* **Слева (Residual-stream L2 norm):** Тепловая карта нормы активаций ($L_2$-нормы) для каждого токена на каждом из 12 слоев GPT-2. Показывает, как изменяется магнитуда скрытых представлений в остаточном потоке (residual stream) по мере прохождения через модель.
* **Справа (Logit lens @ last token):** График вероятности предсказания следующего токена на каждом слое для последнего токена в промпте (`" fruit"`). Этот метод (**Logit Lens**) позволяет заглянуть «в голову» модели на промежуточных слоях, проецируя промежуточные активации напрямую на финальный слой классификации (unembedding matrix). Мы видим, как от первых слоев к последним растет уверенность модели и уточняется предсказываемый токен.


### 3. Open Interactive Demos
Simply open [index.html](file:///home/oleh/Downloads/ai-hack-lab/index.html) or [steering-explained.html](file:///home/oleh/Downloads/ai-hack-lab/steering-explained.html) in any modern browser to interact with the visualizations.
