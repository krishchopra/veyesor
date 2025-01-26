"use client";

import dynamic from "next/dynamic";
import { useRef } from "react";

const ReactPlayer = dynamic(() => import("react-player"), { ssr: false });

export default function VideoPlayer360() {
	const playerRef = useRef<any>(null);

	return (
		<div className="relative w-full h-screen">
			<ReactPlayer
				ref={playerRef}
				url="/sample360.mp4"
				width="100%"
				height="100%"
				playing={true}
				controls={true}
				config={{
					file: {
						attributes: {
							crossOrigin: "anonymous",
							playsInline: true,
						},
					},
				}}
			/>
		</div>
	);
}
