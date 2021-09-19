import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketAddress;
import java.net.InetSocketAddress;
import java.util.ArrayList;

public class Broker extends Node
{
	
	private class worker
	{
		String name;
		InetSocketAddress socketAddress;
		
		worker(String name, int port)
		{
			this.name = name;
			socketAddress = new InetSocketAddress("localhost", port);
		}
		
		public InetSocketAddress getAddress()
		{
			return socketAddress;
		}
		
		public String getName()
		{
			return name;
		}		
	}
	
	static final int DEFAULT_PORT = 50001;

	static final int HEADER_LENGTH = 2;
	static final int TYPE_POS = 0;
	
	static final byte TYPE_UNKNOWN = 0;
	
	static final byte TYPE_MAKE_CONTACT = 1;
	static final int LENGTH_POS = 1;
	static final int PORT_POS = 2;
	static final int PORT_LENGTH = 5;
	static final int NAME_POS = 7;
	
	static final byte TYPE_WORK_DESCRIPTION = 3; // Indicating the data holds work descriptions
	static final byte TYPE_NUMBER_OF_WORKERS = 4;
	static final int WORK_POS = 1;
	
	static final byte TYPE_WITHDRAW = 5;
	
	static final byte TYPE_SEND_RESULTS = 6;
	
	static final byte TYPE_ACK = 2;
	static final int ACKCODE_POS = 1;
	static final byte ACK_ALLOK = 10;
	
	static final byte PORT_SIZE = 5;
	
	@SuppressWarnings("unused")
	private final int MAKE_CONTACT = 0;
	@SuppressWarnings("unused")
	private final int WITHDRAW = 1;
	@SuppressWarnings("unused")
	private final int RESULTS = 2;
	private final int FORWARD_WORK = 3;
	@SuppressWarnings("unused")
	private final int SEND_WORK = 4;
	private final int SEND_WORK_TO_WORKERS = 5;
	private final int SEND_ACK = 9;
	
	boolean continueRunning = true;
	
	ArrayList<worker> workers = new ArrayList<worker>();
	ArrayList<String> workDescriptions = new ArrayList<String>();
	Terminal terminal;
	Thread readerThread;
	reader reader;
	int workersToSendTo;

	Broker(Terminal terminal, int port) {
		try {
			this.terminal = terminal;
			socket = new DatagramSocket(port);
			listener.go();
		}
		catch(java.lang.Exception e) {e.printStackTrace();}
	}
	
	public synchronized void onReceipt(DatagramPacket packet)
	{
		try
		{
			byte[] data;
			
			data = packet.getData();
			terminal.println("Packet Received");
			switch(data[TYPE_POS])
			{
			case TYPE_ACK:
				reader.ackReceived();
				this.notify();
				break;
			case TYPE_NUMBER_OF_WORKERS:
				byte[] content = new byte[data[LENGTH_POS]];
				System.arraycopy(data, HEADER_LENGTH, content, 0, content.length);
				int info = Integer.parseInt(new String(content));
				setNumberOfWorkersToSendTo(info);
				sendAck(packet);
				break;
			case TYPE_WORK_DESCRIPTION:   // send the work description to all clients
				sendAck(packet);
				terminal.println("Received work");
				forwardWork(data); //packet);
				addWorkDescription(data);
				break;
			case TYPE_MAKE_CONTACT:
				contactMade(packet);
				break;
			case TYPE_SEND_RESULTS:
				sendAck(packet);
				break;
			case TYPE_WITHDRAW:
				terminal.println("Request to remove worker");
				removeWorker(packet);
				sendAck(packet);
				break;
			default:
				terminal.println("Unexpected packet" + packet.toString());
			}
		}
		catch(Exception e) {e.printStackTrace();}
	}
	
	public void setNumberOfWorkersToSendTo(int workersToSendTo)
	{
		if(workersToSendTo <= workers.size() && workersToSendTo > 0)
			this.workersToSendTo = workersToSendTo;
		else
			this.workersToSendTo = workers.size();
	}
	
	public void addWorkDescription(byte[] data)
	{
		byte[] workInBytes;
		String work;
		workInBytes = new byte[data[WORK_POS]];
		System.arraycopy(data, HEADER_LENGTH, workInBytes, 0, workInBytes.length);
		work = new String(workInBytes);
		workDescriptions.add(work);
		terminal.println(work);
	}
	
	public void removeWorker(DatagramPacket packet)
	{
		byte[] data = packet.getData();
		byte[] buffer;
		String contentInString;
		int content;
		InetSocketAddress workersAddress;
		
		buffer = new byte[data[LENGTH_POS]];
		System.arraycopy(data, HEADER_LENGTH, buffer, 0, buffer.length);
		contentInString = new String(buffer);
		content = Integer.parseInt(contentInString);
		workersAddress = new InetSocketAddress("localhost", content);
		
		for(int i = 0; i < workers.size(); i++)
		{
			if( workers.get(i).getAddress().toString().compareTo(workersAddress.toString()) == 0 )
			{
				terminal.println( "Removed worker: " + workers.get(i).getName() );
				workers.remove(i);
				break;
			}
		}
		terminal.println("Done");
	}
	
	public synchronized void contactMade(DatagramPacket packet)
	{
		try
		{
			String content;
			String info;
			int port;
			byte[] data;
			byte[] nameBytes;
			byte[] portBytes;
			
			data = packet.getData();
			nameBytes = new byte[data[LENGTH_POS] - PORT_SIZE];
			portBytes = new byte[PORT_SIZE];
			System.arraycopy(data, NAME_POS, nameBytes, 0, nameBytes.length);
			content = new String(nameBytes);
			System.arraycopy(data, HEADER_LENGTH, portBytes, 0, PORT_SIZE);
			info = new String(portBytes);
			
			port = Integer.parseInt(info);
			
			worker newWorker = new worker(content, port);
			workers.add(newWorker);
			
			terminal.println("New worker added: " + content);
			
			sendAck(packet);	
			
			this.wait(2000);
			
			if(workDescriptions.size() > 0)
				sendWorkDescriptions( newWorker.getAddress() );
		} 
		catch(Exception e) {e.printStackTrace();}
	}
	
	public synchronized void forwardWork( byte[] data) //DatagramPacket packet)
	{
		try
		{
			terminal.println("Forwarding work");
			
			for(int i = 0; i < workersToSendTo; i++)
			{
				InetSocketAddress dstAddress = workers.get(i).getAddress();
				reader = new reader(FORWARD_WORK, dstAddress, socket, data); //packet);
				readerThread = new Thread( reader );
				readerThread.start();
				terminal.println("Work sent to " + (i + 1) + " of " + workersToSendTo + " workers");
			}
		}
		catch(Exception e) {e.printStackTrace();}
	}
	
	public synchronized void sendAck(DatagramPacket packet)
	{
		try
		{
			SocketAddress dstAddress = packet.getSocketAddress();
			reader = new reader(SEND_ACK, dstAddress, socket);
			readerThread = new Thread( reader );
			readerThread.start();
		}
		catch(Exception e) {e.printStackTrace();}
	}
	
	public synchronized void sendWorkDescriptions(InetSocketAddress dstAddress) throws Exception
	{
		for(int i = 0; i < workDescriptions.size(); i++)
		{
			terminal.println("Sending work");
			reader = new reader(SEND_WORK_TO_WORKERS, dstAddress, socket, workDescriptions.get(i));
			readerThread = new Thread( reader );
			readerThread.start();
			terminal.println("Work Sent");
		}
	}
	
	public synchronized void run()
	{
		try
		{
			terminal.println("Running...");
			Boolean continueLoop = true;
			while(continueLoop == true)
			{
				this.wait();
			}
		}
		catch(Exception e) {e.printStackTrace();}
	}
	
	public static void main(String[] args)
	{
		try {
			Terminal terminal = new Terminal("Broker");
			(new Broker(terminal, DEFAULT_PORT)).run();
			terminal.println("Program completed");
		}
		catch(java.lang.Exception e) {e.printStackTrace();}
	}
}
