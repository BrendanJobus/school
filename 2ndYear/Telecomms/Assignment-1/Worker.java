import java.net.DatagramSocket;
import java.net.DatagramPacket;
import java.net.InetSocketAddress;

public class Worker extends Node
{
	static final int DEFAULT_SRC_PORT = 50002; // Port of the worker
	static final int DEFAULT_DST_PORT = 50001; // Port of the Broker
	static final String DEFAULT_DST_NODE = "localhost";	// Name of the host for the server

	static final int HEADER_LENGTH = 2; // Fixed length of the header
	static final int TYPE_POS = 0; // Position of the type within the header

	static final byte TYPE_UNKNOWN = 0;

	static final byte TYPE_MAKE_CONTACT = 1; // Indicating a string payload
	static final int LENGTH_POS = 1;
	static final int PORT_POS = 2;
	static final int PORT_LENGTH = 5;
	static final int NAME_POS = 7;
	
	static final byte TYPE_WORK_DESCRIPTION = 3; // Indicating the data holds work descriptions
	static final byte TYPE_NUMBER_OF_WORKERS = 4;
	static final int WORK_POS = 2;
	
	static final byte TYPE_WITHDRAW = 5;
	
	static final byte TYPE_SEND_RESULTS = 6;
	
	static final byte TYPE_ACK = 2;   // Indicating an acknowledgement
	static final int ACKCODE_POS = 1; // Position of the acknowledgement type in the header
	static final byte ACK_ALLOK = 10; // Indicating that everything is ok

	static final byte PORT_SIZE = 5;

	private final int MAKE_CONTACT = 0;
	private final int WITHDRAW = 1;
	private final int RESULTS = 2;
	@SuppressWarnings("unused")
	private final int FORWARD_WORK = 3;
	@SuppressWarnings("unused")
	private final int SEND_WORK = 4;
	@SuppressWarnings("unused")
	private final int SEND_WORK_TO_WORKERS = 5;
	private final int SEND_ACK = 9;
	
	boolean workToAccept = false;
	String name;
	int srcPort;
	Terminal terminal;
	InetSocketAddress dstAddress;
	Thread readerThread;
	reader reader;
	
	Worker(Terminal terminal, String dstHost, int dstPort, int srcPort) {
		try {
			this.terminal = terminal;
			dstAddress = new InetSocketAddress(dstHost, dstPort);
			socket = new DatagramSocket(srcPort);
			this.srcPort = srcPort;
			listener.go();
		}
		catch(java.lang.Exception e) {e.printStackTrace();}
	}
	
	public synchronized void onReceipt(DatagramPacket packet) {
		try
		{
			byte[] data;
	
			data = packet.getData();
			terminal.println("Packet Received");
			switch(data[TYPE_POS]) {
			case TYPE_ACK:
				reader.ackReceived();
				this.notify();
				break;
			case TYPE_WORK_DESCRIPTION:
				printDescriptions(data);
				sendAck(packet);
				break;
			case TYPE_MAKE_CONTACT:
				break;
			default:
				terminal.println("Unexpected packet" + packet.toString());
			}
		}
		catch(Exception e) {e.printStackTrace();}
	}
	
	public synchronized void printDescriptions(byte[] data)
	{
		try {
			String content;
			byte[] info;
						
			info = new byte[data[LENGTH_POS]];
			System.arraycopy(data, HEADER_LENGTH, info, 0, info.length);
			content = new String(info);
			terminal.println(content);
		}
		catch(Exception e) {e.printStackTrace();}
	}
	
	public synchronized void sendResults(String message) throws Exception
	{
		terminal.println("Sending Results to broker");
		reader = new reader(RESULTS, dstAddress, socket, message);
		readerThread = new Thread( reader );
		readerThread.start();
		this.wait();
		terminal.println("Results successfully sent to broker");
	}
	
	public synchronized void makeContact() throws Exception // send a message containing your name to the server and thus asking for work
	{
		terminal.println("Contacting broker...");
		reader = new reader(MAKE_CONTACT, dstAddress, socket, srcPort, name);
		readerThread = new Thread( reader );
		readerThread.start();
		this.wait();
		terminal.println("Contact made");
	}
	
	public synchronized void withdraw()
	{
		try
		{
			terminal.println("Sending withdrawl");
			reader = new reader(WITHDRAW, dstAddress, socket, srcPort);
			readerThread = new Thread( reader );
			readerThread.start();
			this.wait();
			terminal.println("You have been withdrawn from work");
		}
		catch(Exception e) {e.printStackTrace();}
	}

	public synchronized void sendAck(DatagramPacket packet)
	{
		try
		{
			reader = new reader(SEND_ACK, dstAddress, socket, packet);
			readerThread = new Thread( reader );
			readerThread.start();
		}
		catch(Exception e) {e.printStackTrace();}
	}
	
	public synchronized void run()
	{
		try
		{
			name = terminal.read("Name: ");
			makeContact();
			terminal.println("Printing work when available");
			String command;
			Boolean continueLoop = true;
			while(continueLoop == true)
			{
				this.wait();
				command = terminal.read("\"Withdraw\" from work or send results: ");
				if(command.equalsIgnoreCase("withdraw"))
				{
					terminal.println("Withdrawing from work");
					withdraw();
					continueLoop = false;
				}
				else
				{
					sendResults(command);
				}
			}
		}
		catch(Exception e) {e.printStackTrace();}
	}
	
	public static void main(String[] args) {
		try {
			Terminal terminal = new Terminal("Worker");
			(new Worker(terminal, DEFAULT_DST_NODE, DEFAULT_DST_PORT, DEFAULT_SRC_PORT)).run();
			terminal.println("Program completed");
		} catch(java.lang.Exception e) {e.printStackTrace();}
	}
}