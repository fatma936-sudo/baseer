"""Baseer agents — one module per model we call:
  agent1_reasoning  Fanar-C-2-27B   (the brain / decision-maker)
  agent2_voice      Aura ASR + TTS  (hear + speak Arabic)
  agent3_vision     Fanar-Oryx      (identify + localize items)
  agent4_grasp      SmolVLA         (grasp + deliver on the SO-100)
  agent5_yolo       YOLO-World      (alternative local eyes)
  orchestrator      the ReAct loop that drives agent1 to choose tools
"""
