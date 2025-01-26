import VideoPlayer360 from "./components/VideoPlayer360";

export default function Home() {
	return (
		<main className="min-h-screen">
			<VideoPlayer360 wsUrl="ws://your-backend-server/video-stream" />
		</main>
	);
}
