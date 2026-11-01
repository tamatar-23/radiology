import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Multimodal Agentic Radiology Assistant — Chest X-Ray Research Prototype',
  description:
    '4th-Year Capstone Research Prototype for Chest X-Ray Preprocessing, Multi-Label Classification, and Agentic RAG Report Synthesis.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 text-slate-900 antialiased">
        {children}
      </body>
    </html>
  );
}
