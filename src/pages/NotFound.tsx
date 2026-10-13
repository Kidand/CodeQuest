/** 404 / 未知资源 */
import { Link } from 'react-router'
import { Compass } from 'lucide-react'
import { EmptyState } from '@/components/game/EmptyState'

export default function NotFound({ title = '这里什么也没有', children }: { title?: string; children?: React.ReactNode }) {
  return (
    <div className="py-10">
      <EmptyState
        icon={Compass}
        title={title}
        action={
          <>
            <Link to="/" className="inline-flex h-10 items-center rounded-xl bg-accent px-4 text-sm font-medium text-accent-fg shadow-sm hover:bg-accent-hover">
              回到首页
            </Link>
            <Link to="/track/algo" className="inline-flex h-10 items-center rounded-xl border border-border px-4 text-sm font-medium hover:bg-surface-2">
              打开闯关地图
            </Link>
          </>
        }
      >
        {children ?? '页面不存在，或者对应的章节 / 关卡还没有上线。'}
      </EmptyState>
    </div>
  )
}
