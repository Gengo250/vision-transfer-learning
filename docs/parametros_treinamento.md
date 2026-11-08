# DeepPCB: detecção de defeitos e transfer learning

Este documento descreve [deep_pcb_transfer_learning.ipynb](../notebooks/deep_pcb_transfer_learning.ipynb). O objetivo é comparar três inicializações e políticas de treinamento do mesmo detector. Consulte o [ambiente compartilhado](environment.md) para instalação e execução.

## Dados e integridade

O [DeepPCB oficial](https://github.com/tangsanli5201/DeepPCB) reúne 1.500 pares de imagens de placas, com uma imagem inspecionada e um template alinhado sem defeitos. O dataset usado anteriormente neste projeto tinha 10.013 anotações; esses totais dependem da cópia e não foram recontados nesta revisão. A resolução informada é 640 × 640 pixels. Uma cópia foi disponibilizada no [Google Drive](https://drive.google.com/drive/folders/1mdTe8Ez5r7KCi5ryaBf_onUCZJHOlPqH?usp=drive_link).

A pasta esperada contém `trainval.txt`, `test.txt` e diretórios de grupos. Cada linha dos splits aponta para uma imagem-base e seu arquivo de anotações; os arquivos de imagem correspondentes terminam em `_test.jpg` e `_temp.jpg`. Cada linha da anotação contém `x1 y1 x2 y2 classe`, com caixas XYXY em pixels absolutos. Os IDs de 1 a 6 representam, respectivamente, `open`, `short`, `mousebite`, `spur`, `copper` e `pin-hole`; 0 é fundo.

O padrão é `data/DeepPCB/PCBData/` localmente e `MyDrive/vision-transfer-learning/dataset/DeepPCB/PCBData/` no Colab. `CONFIG['dataset_dir']` é relativo a `data/` ou `dataset/`, respectivamente; caminhos absolutos também podem ser usados. O notebook não baixa nem copia os dados.

O leitor percorre os arquivos listados, conta imagens e anotações, mostra a distribuição por classe e os grupos compartilhados; também exibe uma imagem inspecionada com caixas ao lado do template. A ausência da pasta gera erro e arquivos ilegíveis ou anotações malformadas podem interromper a leitura. **Não há auditoria completa automatizada** de dimensões, duplicatas, limites das caixas, existência de todos os templates ou integridade dos pixels. Confira a origem e a integridade da cópia antes do treino.

## Divisão e preparação

`trainval.txt` é dividido por grupos inteiros, buscando cobrir as seis classes nos dois lados e aproximar 15% das imagens na validação; a semente desempata candidatos. `test.txt` permanece como teste oficial. Na cópia anteriormente auditada, a divisão foi 850/150/500 imagens para treino/validação/teste. As contagens reais são impressas ao executar o notebook.

Treino e validação não compartilham grupos, mas o teste oficial compartilha grupos com o desenvolvimento. Assim, o teste não demonstra generalização para placas ou grupos independentes. O template é apenas visualizado: o detector recebe **somente** a imagem `_test.jpg`, convertida para RGB e tensor `float32` em `[0, 1]`. Não há data augmentation implementado. O modelo faz normalização interna e redimensiona, por padrão, com `min_size=800`, `max_size=1333`; a ampliação não cria novos detalhes. Testar outra resolução é uma proposta, mantendo-a igual em A/B/C para comparação.

## Modelo e estratégias

O modelo é o [`fasterrcnn_resnet50_fpn_v2`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html), não um Mask R-CNN. O ResNet-50 extrai características, a FPN fornece escalas múltiplas, a RPN propõe regiões e as cabeças RoI classificam e refinam caixas. Em inferência, cada imagem produz caixas XYXY, classes e scores; não há máscaras. Os alvos incluem caixas, rótulos, `image_id`, área e `iscrowd`.

`FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT` corresponde a pesos COCO; a cabeça `FastRCNNPredictor` é substituída por uma com sete classes, incluindo fundo. O notebook implementa uma estratégia por rodada:

| `experiment` | Inicialização | Parâmetros treinados |
| --- | --- | --- |
| `A` | Aleatória, sem COCO nem ImageNet | Todo o detector |
| `B` | COCO, com preditor novo | RPN e cabeças; ResNet-50 e FPN congelados |
| `C` | COCO, com preditor novo | Todo o detector |

Em B, o backbone também permanece em `eval()` durante o treino para não atualizar estatísticas de normalização. A mede a referência sem transferência; B mede o aproveitamento de características fixas; C permite adaptá-las ao domínio. Nenhuma estratégia tem superioridade comprovada aqui. Para comparação controlada, mantenha os demais parâmetros, dados e divisão iguais; repetir sementes permite estimar variação.

## Configuração e treinamento

Valores atuais da célula `CONFIG`:

| Parâmetro | Padrão | Função |
| --- | --- | --- |
| `seed` | `42` | Sementes Python, NumPy e PyTorch e desempate dos grupos |
| `dataset_dir` | `DeepPCB/PCBData` | Pasta dos dados relativa à raiz de dados do ambiente |
| `experiment` | `C` | Estratégia A, B ou C |
| `epochs`, `patience` | `20`, `5` | Máximo de épocas e parada após épocas sem melhora |
| `batch_size`, `num_workers` | `2`, `0` | Imagens por batch e processos de leitura |
| `amp` | `True` | Precisão mista com GradScaler apenas em CUDA |
| `learning_rate`, `momentum`, `weight_decay` | `0.001`, `0.9`, `0.0005` | SGD nos parâmetros treináveis |
| `lr_step_size`, `lr_gamma` | `7`, `0.1` | StepLR reduz a taxa por fator 0,1 a cada sete épocas |
| `score_threshold`, `iou_threshold` | `0.5`, `0.5` | Limiares das métricas pontuais e matriz de confusão |
| `run_training`, `evaluate_test` | `True`, `False` | Iniciar treino e, opcionalmente, avaliar o teste |
| `checkpoint_path` | `None` | Checkpoint para teste sem treino na mesma execução |

Após mudar `CONFIG`, reexecute sua célula; para `batch_size` ou `num_workers`, recrie os DataLoaders. Um `collate_fn` permite números distintos de alvos por imagem e somente o treino embaralha os dados. A perda de treino é a soma das perdas retornadas pelo TorchVision: classificação e regressão de caixas da RPN e das cabeças de detecção. Uma perda não finita interrompe a rodada. O checkpoint guarda também otimizador, scheduler e scaler, mas o notebook não implementa retomada de treino.

O registro anterior de parâmetros relatava, para uma RTX 3050 Laptop de 4 GB com batch 2 e imagens a 800 px, passos de aproximadamente 725 ms em FP32 e 364 ms com AMP, com picos de 2,7 e 1,7 GB, respectivamente. Também relatava leitura de imagem em cerca de 3,5 ms. Esses números históricos não têm protocolo ou artefato de medição associado nesta documentação e não foram verificados nesta revisão; não devem ser generalizados para outras máquinas ou rodadas.

Os padrões do detector que não são configurados no notebook incluem âncoras de 32 a 512 pixels por nível, normalização com médias e desvios ImageNet, `box_score_thresh=0.05`, `box_nms_thresh=0.5` e no máximo 100 detecções por imagem. Esses padrões dependem da versão do TorchVision; o limite pode afetar imagens com muitos defeitos.

## Avaliação e artefatos

A cada época, a validação calcula mAP de caixas no protocolo COCO via TorchMetrics/pycocotools: `map_50` e `map_50_95`. A mAP usa os scores retornados pelo detector, sem o filtro adicional `score_threshold`; o detector ainda aplica seus próprios filtros e NMS. O benchmark DeepPCB original usa IoU 0,33, portanto seus números não são diretamente comparáveis aos deste protocolo.

Precisão, recall e F1 são micro, usando confiança e IoU de `CONFIG` e casamento guloso por score que exige a classe correta. A matriz de confusão casa caixas por posição independentemente da classe, com linhas reais e colunas previstas, incluindo fundo. O melhor checkpoint é escolhido pelo maior `map_50_95` de validação; o early stopping monitora a mesma métrica. Não há perda de validação calculada.

Para reservar o teste à avaliação final, escolha a estratégia com validação e só depois use `evaluate_test=True`. Com `run_training=False`, informe `checkpoint_path`; com treino na mesma execução, usa-se o melhor checkpoint criado. Caminhos relativos partem da raiz do projeto. Carregue apenas checkpoints confiáveis: `torch.load` usa `weights_only=False`.

Cada rodada cria `runs/results/<data_hora>_<experimento>/` no ambiente ativo com `config.json`, `split.csv`, `history.csv`, `checkpoints/best.pt`, curvas individuais e painel em `figures/`, matriz de validação em `.png` e `.npy` e, se solicitado, `test_metrics.json` e matriz de teste. A visualização implementada mostra exemplos anotados e matrizes; não há rotina de visualização de predições sobre a imagem. Sementes ajudam a reproduzir divisões e inicialização, mas não garantem determinismo em todos os dispositivos.

Os resultados dependem do dataset e da execução; nenhuma métrica de desempenho de treinamento completo é afirmada nesta documentação.

[Índice](README.md).
