# Índice da documentação

Esta documentação descreve os notebooks presentes no repositório. Os valores de configuração devem ser conferidos na célula `CONFIG` antes de cada rodada.

## Ambiente geral

- [Ambiente e execução](environment.md): Python, uv, CPU/GPU, JupyterLab e Google Colab.

## Experimento DeepPCB

- [DeepPCB: dados, modelo, transferência e avaliação](parametros_treinamento.md): Faster R-CNN, divisões, parâmetros e métricas. Inclui o desenho experimental das estratégias A, B e C: o que cada uma mede, o mecanismo, os sinais para observar nas curvas, o que cada comparação isola, as ameaças à validade e o protocolo sugerido.
- [Notebook DeepPCB](../notebooks/deep_pcb_transfer_learning.ipynb).

## Experimento HRSID

- [HRSID: dados, modelo, transferência e avaliação](parametros_hrsid.md): Mask R-CNN, anotações de instância, divisões, parâmetros e métricas. Inclui o desenho experimental de `frozen_backbone`, `partial_finetune` (com a varredura de `trainable_layers`) e `full_finetune`, o efeito do domínio SAR em cada nível da rede, as comparações, as ameaças à validade, o protocolo sugerido e a matriz de confusão.
- [Notebook HRSID](../notebooks/hrsid_mask_rcnn_transfer_learning.ipynb).

[Voltar ao início](../README.md).
