import '@/styles/globals.css'
import type { AppProps } from 'next/app'
import { useState } from 'react'
import Head from 'next/head'

export default function App({ Component, pageProps }: AppProps) {
  return (
    <>
      <Head>
        <title>深度推定・3D可視化アプリ</title>
        <meta name="description" content="深度推定と3D可視化を行うWebアプリケーション" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <Component {...pageProps} />
    </>
  )
}