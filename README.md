# 🚀 Dynamic Evidence Control: Meta-Policy Learning for Adaptive Uncertainty Calibration

📄 Link to the paper: [Paper Address](#)

---

## 🧠 Project Description

Uncertainty quantification (UQ) is essential in risk-sensitive applications such as medical diagnosis 🏥 and autonomous driving 🚗. It enhances prediction reliability, improves interpretability, and supports safe decision-making.

Evidence Deep Learning (EDL) offers a popular framework for UQ by modeling prediction uncertainty via the Dirichlet distribution, enabling joint prediction and confidence estimation in a single forward pass. However, traditional EDL methods suffer from:

- ❗ Sensitivity to fixed hyperparameters like KL regularization coefficient and Dirichlet prior,
- 🔧 The need for manual tuning,
- 📉 Poor adaptability to shifting data distributions and training dynamics.

To overcome these limitations, we propose a novel meta-learning-based regularization framework:

### ✨ Meta-Policy Controller (MPC)

MPC introduces a state-aware policy network that dynamically configures the KL coefficient λ and Dirichlet prior α₀ based on the model's training state. Specifically:

- 📈 The policy observes both short-term (batch-level) and long-term (epoch-level) model statistics.
- 🎯 It generates hyperparameters that control the strength and direction of regularization.
- 🏆 A reward function balancing classification accuracy and uncertainty calibration guides the policy updates.

Our method enables EDL models to learn more effectively in dynamic environments, achieving:

- ✅ Higher classification accuracy,
- 📊 Better uncertainty calibration (lower ECE and MUE),
- 🛡️ Improved performance in out-of-distribution (OOD) settings and sample rejection tasks.

For a visual overview, check out our method diagram:  
📷 ![Overview](https://github.com/xi-xiaoran/Meta-Policy-Controller/blob/main/Pictures/overview.png)

---

## 📦 Installation

This project requires the following Python libraries:

```bash
matplotlib==3.5.3
numpy==1.21.5
Pillow==11.2.1
scikit_learn==1.0.2
scipy==1.7.3
seaborn==0.13.2
torch==1.13.1
torchvision==0.14.1
tqdm==4.66.1
```
