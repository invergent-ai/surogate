import React from 'react'
import useBaseUrl from '@docusaurus/useBaseUrl'
import {useColorMode} from '@docusaurus/theme-common';

export default function Index({src, darkSrc, caption}) {
    const isDarkTheme = useColorMode().colorMode === "dark";
    const darkImageSrc = darkSrc || src;
    return (
        <figure style={{padding: 20}}>
            <img src={isDarkTheme ? useBaseUrl(darkImageSrc) : useBaseUrl(src)} alt={caption}/>
            { caption &&
            <figcaption style={{textAlign: 'center', fontWeight: 200, marginTop: '1rem'}}>{`Figure: ${caption}`}</figcaption>
            }
        </figure>
    )
}
