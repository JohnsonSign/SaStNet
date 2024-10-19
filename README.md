# SaStNet

Saliency-aware Spatio-temporal Modeling for Action Recognition on Unmanned Aerial Vehicles

Action recognition on unmanned aerial vehicles (UAVs) must cope with complex backgrounds and focus on small targets. Existing methods usually use additional detectors to extract objects in each frame, and use the object sequence within boxes as the network input. However, for training, they rely on additional detection annotations, and for inference, the multi-stage paradigm increases the burden of deployment on UAV terminals. Therefore, we propose a saliency-aware spatiotemporal network (SaStNet) for UAV-based action recognition in an end-to-end manner. Specifically, the short-term and long-term motion information are captured progressively. For short-term modeling, a saliency-guided enhancement module is designed to learn attention scores for weighting the original features aggregated within neighboring frames. For long-term modeling, informative regions are first adaptively concentrated using a saliency-guided aggregation module. Then, a spatio-temporal decoupling attention mechanism is designed to focus on spatially salient regions and capture temporal relationships within all frames. Integrating these modules into classical backbones encourages the network to focus on moving targets, reducing interference from background noises.

# Train and test

bash script_train_val.sh
