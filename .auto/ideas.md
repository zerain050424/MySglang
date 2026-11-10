# 未执行的优化想法（新会话，随实验补充）

- 多级队列仲裁替代 Min-Max 水填充：候选方向包括按队列「边际命中率增益」分配淘汰量、
  按周转率自适应配额、protected 段降级（demotion）。先用 profile 数据确认现有仲裁
  在哪个 cell 上损失命中率，再动手。
- 带内「会话活跃度」判据：用 SessionState.gaps 中位数自校准闲置判定（避免固定秒数调参）。
- 保护段降级：share_sessions 只增不减、protected 只进不出；长期不复用的模板应可降级。
- run 改动前先确认这些想法与 MixAgentic（基线已含）是否重叠。
