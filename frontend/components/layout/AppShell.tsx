'use client'

import Sidebar from './Sidebar'

export default function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative min-h-screen">
      <Sidebar />
      <div className="lg:pl-64">
        <main className="mx-auto max-w-7xl px-5 pb-16 pt-20 lg:px-10 lg:pt-10">{children}</main>
      </div>
    </div>
  )
}
