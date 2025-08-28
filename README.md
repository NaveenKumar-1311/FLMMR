# FLMMR
Federated Learning Minimal Model Replacement Attack Using Optimal Transport: An Attacker Perspective published in IEEE Transactions on Information Forensics and Security

Abstract: Federated learning (FL) has emerged as a powerful collaborative learning approach that enables client devices to
train a joint machine learning model without sharing private data. However, the decentralized nature of FL makes it highly
vulnerable to adversarial attacks from multiple sources. There are diverse FL data poisoning and model poisoning attack
methods in the literature. Nevertheless, most of them focus only on the attack’s impact and do not consider the attack budget and attack visibility. These factors are essential to effectively comprehend the adversary’s rationale in designing an attack. Hence, our work highlights the significance of considering these factors by providing an attacker perspective in designing an attack with a low budget, low visibility, and high impact. Also, existing attacks that use total neuron replacement and randomly selected neuron replacement approaches only cater to these factors partially. Therefore, we propose a novel federated learning minimal model replacement attack (FL-MMR) that uses optimal transport (OT) for minimal neural alignment between a surrogate poisoned model and the benign model. Later, we optimize the attack budget in a three-fold adaptive fashion by considering critical learning periods and introducing the replacement map. In addition, we comprehensively evaluate our attack under three threat scenarios using three large-scale datasets: GTSRB, CIFAR10, and EMNIST. 


Citation  
If you use this work, please cite:  
```
@ARTICLE{10802950,
  author={Naveen Kumar, K. and Krishna Mohan, C. and Reddy Cenkeramaddi, Linga},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Federated Learning Minimal Model Replacement Attack Using Optimal Transport: An Attacker Perspective}, 
  year={2025},
  volume={20},
  number={},
  pages={478-487},
  keywords={Adaptation models;Servers;Data models;Computational modeling;Federated learning;Training;Neurons;Costs;Accuracy;Toxicology;Federated learning;model replacement attack;attack impact;adaptive attack budget;attack visibility},
  doi={10.1109/TIFS.2024.3516555}}
```
