/**
 * 逐帧推导：KL 约束奖励最大化 → 闭式解 → 反解奖励 → 代入 Bradley-Terry → Z(x) 相消 → DPO loss。
 * 每一帧给出一条等式 + “这一步做了什么 / 为什么成立”。
 */
import { StepControls, Tex, VizFrame, useStepper, toneColor, toneSoft } from '@/viz'
import { cn } from '@/lib/utils'

interface Frame {
  title: string
  tex: string[]
  why: string
}

const FRAMES: Frame[] = [
  {
    title: '① RLHF 的目标',
    tex: [
      String.raw`\max_{\pi}\ \mathbb{E}_{x\sim\mathcal D,\,y\sim\pi(\cdot|x)}\big[r(x,y)\big]`,
      String.raw`-\ \beta\,\mathbb{D}_{\mathrm{KL}}\big[\pi(y|x)\,\|\,\pi_{\mathrm{ref}}(y|x)\big]`,
    ],
    why: 'PPO 阶段优化的就是它：奖励越高越好，但不能离 SFT 模型 π_ref 太远（否则会钻奖励模型的空子）。',
  },
  {
    title: '② 合并成一个期望',
    tex: [
      String.raw`=\ \mathbb{E}_{y\sim\pi}\Big[r(x,y)-\beta\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)}\Big]`,
      String.raw`=\ -\beta\,\mathbb{E}_{y\sim\pi}\Big[\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)\,e^{r(x,y)/\beta}}\Big]`,
    ],
    why: '对固定的 x 看：KL 本身就是 E_π[log π/π_ref]；把 r 写成 β·log e^{r/β} 塞进同一个对数里。',
  },
  {
    title: '③ 凑出一个分布 π*',
    tex: [
      String.raw`Z(x)=\sum_{y}\pi_{\mathrm{ref}}(y|x)\,e^{r(x,y)/\beta}`,
      String.raw`\pi^*(y|x)=\frac{\pi_{\mathrm{ref}}(y|x)\,e^{r(x,y)/\beta}}{Z(x)}`,
      String.raw`\text{目标}=-\beta\,\mathbb{D}_{\mathrm{KL}}[\pi\|\pi^*]+\beta\log Z(x)`,
    ],
    why: '分母不是概率分布，除以配分函数 Z(x) 归一化。Z(x) 只依赖 x、不依赖 π，是常数。',
  },
  {
    title: '④ 闭式最优解',
    tex: [
      String.raw`\mathbb{D}_{\mathrm{KL}}[\pi\|\pi^*]\ge 0,\ \ \pi=\pi^*\ \text{时取等}`,
      String.raw`\Rightarrow\ \pi^*(y|x)\propto\pi_{\mathrm{ref}}(y|x)\exp\!\big(r(x,y)/\beta\big)`,
    ],
    why: 'Gibbs 不等式：KL 非负。所以最优策略就是把 π_ref 按 exp(r/β) 重新加权。问题：Z(x) 要对所有可能的回答求和，算不出来。',
  },
  {
    title: '⑤ 反过来：用策略表示奖励',
    tex: [String.raw`r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\mathrm{ref}}(y|x)}+\beta\log Z(x)`],
    why: '对④取对数移项。关键视角转换：不再“先有奖励再求策略”，而是让策略本身隐式地定义奖励。',
  },
  {
    title: '⑥ 代入 Bradley-Terry，Z(x) 相消',
    tex: [
      String.raw`p(y_w\succ y_l|x)=\sigma\big(r(x,y_w)-r(x,y_l)\big)`,
      String.raw`=\sigma\Big(\beta\log\frac{\pi^*(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}+\cancel{\beta\log Z(x)}`,
      String.raw`-\ \beta\log\frac{\pi^*(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}-\cancel{\beta\log Z(x)}\Big)`,
    ],
    why: 'BT 模型只看奖励之差。同一个 prompt x 下两条回答的 β·log Z(x) 完全相同，一减就没了——这正是 DPO 只能用成对（同 prompt）数据的原因。',
  },
  {
    title: '⑦ 参数化 + 最大似然 = DPO loss',
    tex: [
      String.raw`\mathcal L_{\mathrm{DPO}}=-\,\mathbb{E}_{(x,y_w,y_l)}\Big[\log\sigma\Big(`,
      String.raw`\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\Big)\Big]`,
    ],
    why: '把 π* 换成可训练的 π_θ，对偏好数据做负对数似然。等价于用“隐式奖励” r̂ = β·log(π_θ/π_ref) 训练一个 BT 奖励模型——没有显式 RM，也不用采样。',
  },
]

export function DerivationViz() {
  const st = useStepper(FRAMES, { intervalMs: 2600 })
  const f = st.frame
  return (
    <VizFrame title="七步推出 DPO：看 Z(x) 怎么消掉" caption={<>{f.why}</>} controls={<StepControls stepper={st} />}>
      <div className="grid gap-4 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,3fr)]">
        <ol className="flex flex-wrap gap-1.5 @3xl:flex-col">
          {FRAMES.map((fr, i) => (
            <li key={fr.title}>
              <button
                onClick={() => st.setI(i)}
                className={cn('rounded-lg border px-2 py-1 text-left text-[12.5px] transition-colors', i === st.i ? 'font-semibold text-fg' : 'text-muted')}
                style={{
                  borderColor: i === st.i ? toneColor('accent') : 'var(--border)',
                  background: i === st.i ? toneSoft('accent', 14) : i < st.i ? 'var(--surface-2)' : 'var(--surface)',
                }}
              >
                {fr.title}
              </button>
            </li>
          ))}
        </ol>
        <div className="flex min-h-[132px] min-w-0 flex-col justify-center rounded-xl border border-border bg-surface px-3 py-4">
          <div className="mb-2 text-[13px] font-semibold text-fg">{f.title}</div>
          <div className="thin-scroll overflow-x-auto text-[12.5px] @md:text-[15px]">
            {f.tex.map((t, k) => (
              <div key={k} className="w-max min-w-full py-1.5 text-center">
                <Tex>{String.raw`\displaystyle ` + t}</Tex>
              </div>
            ))}
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
