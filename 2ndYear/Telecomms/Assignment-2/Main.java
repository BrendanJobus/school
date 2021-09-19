public class Main 
{

	public static void main(String[] args) 
	{
		try
		{
			Terminal terminal1 = new Terminal("Controller");
			Terminal terminal2 = new Terminal("Router 1");
			Terminal terminal3 = new Terminal("Router 2");
			Terminal terminal4 = new Terminal("Router 3");
			Terminal terminal5 = new Terminal("Router 4");
			Terminal terminal6 = new Terminal("Sender");
			Terminal terminal7 = new Terminal("Receiver");
			
			Controller controller = new Controller(terminal1, 50000, 50001, 50002,
					50003, 50004, 50005, 50006);
			Router router1 = new Router(terminal2, 50002, 50001, 50003, 50000);
			Router router2 = new Router(terminal3, 50003, 50002, 50004, 50000);
			Router router3 = new Router(terminal4, 50004, 50003, 50005, 50000);
			Router router4 = new Router(terminal5, 50005, 50004, 50006, 50000);
			Sender sender = new Sender(terminal6, 50001, 50002);
			Receiver receiver = new Receiver(terminal7, 50006, 50005);
			
			Thread controllerThread = new Thread( controller );
			Thread router1Thread = new Thread( router1 );
			Thread router2Thread = new Thread( router2 );
			Thread router3Thread = new Thread( router3 );
			Thread router4Thread = new Thread( router4 );
			Thread senderThread = new Thread( sender );
			Thread receiverThread = new Thread( receiver );
			
			controllerThread.start();
			router1Thread.start();
			router2Thread.start();
			router3Thread.start();
			router4Thread.start();
			senderThread.start();
			receiverThread.start();
		}
		catch(java.lang.Exception e)
		{
			e.printStackTrace();
		}
	}
}
