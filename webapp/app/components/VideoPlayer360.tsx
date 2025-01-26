"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import type ReactPlayer from "react-player";

const ReactPlayerComponent = dynamic(() => import("react-player"), {
	ssr: false,
});

interface VideoPlayer360Props {
	signalingServer: string;
}

export default function VideoPlayer360({
	signalingServer,
}: VideoPlayer360Props) {
	const playerRef = useRef<ReactPlayer>(null);
	const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
	const [streamUrl, setStreamUrl] = useState<MediaStream | null>(null);

	useEffect(() => {
		const configuration: RTCConfiguration = {
			iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
		};

		const peerConnection = new RTCPeerConnection(configuration);
		peerConnectionRef.current = peerConnection;

		peerConnection.ontrack = (event) => {
			const [stream] = event.streams;
			setStreamUrl(stream);
		};

		const ws = new WebSocket(signalingServer);

		ws.onmessage = async (event) => {
			const message = JSON.parse(event.data);

			switch (message.type) {
				case "offer":
					await peerConnection.setRemoteDescription(
						new RTCSessionDescription(message)
					);
					const answer = await peerConnection.createAnswer();
					await peerConnection.setLocalDescription(answer);
					ws.send(JSON.stringify(answer));
					break;

				case "ice-candidate":
					if (message.candidate) {
						await peerConnection.addIceCandidate(
							new RTCIceCandidate(message)
						);
					}
					break;
			}
		};

		peerConnection.onicecandidate = (event) => {
			if (event.candidate) {
				ws.send(
					JSON.stringify({
						type: "ice-candidate",
						candidate: event.candidate,
					})
				);
			}
		};

		return () => {
			ws.close();
			peerConnection.close();
		};
	}, [signalingServer]);

	if (!streamUrl) {
		return <div>Connecting to video stream...</div>;
	}

	return (
		<div className="relative w-full h-screen">
			<ReactPlayerComponent
				ref={playerRef}
				url={streamUrl}
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
